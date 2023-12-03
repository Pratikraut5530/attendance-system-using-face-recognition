import numpy as np
import pandas as pd
import cv2
import redis
from sklearn.metrics import pairwise
from insightface.app import FaceAnalysis

# Connect to Redis Client
redis_host = "redis-13914.c322.us-east-1-2.ec2.cloud.redislabs.com"
redis_port = 13914
redis_password = "h9fng3OkGwuY1FxPHguTBZ8rVt6uCdrx"
r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password)

# Check Redis connection
try:
    print(r.ping())
except redis.ConnectionError as e:
    print(f"Error: Unable to connect to Redis server - {e}")

# Retrieve data from the database
def retrieve_data(name='academy:register'):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role', 'facial_features']
    retrieve_df[['Name', 'Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name', 'Role', 'facial_features']]

# Configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model\models', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# ML search algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name', 'role'], thresh=0.5):
    # Cosine similarity-based search algorithm
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    
    return person_name, person_role

# Face prediction function
def face_prediction(test_image, dataframe, feature_column, name_role=['Name', 'Role'], thresh=0.5):
    results = faceapp.get(test_image)
    test_copy = test_image.copy()

    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(dataframe, feature_column, test_vector=embeddings,
                                                       name_role=name_role, thresh=thresh)

        color = (0, 0, 255) if person_name == "Unknown" else (0, 255, 0)
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
        text_gen = person_name
        cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return test_copy
