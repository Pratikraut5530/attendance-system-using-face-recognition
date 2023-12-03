import streamlit as st  
st.set_page_config('Prediction',layout='wide')
st.subheader('Real Time Attendence System')


## Retrive Data from Redis Database

import facerec
redis_face_db = facerec.retrive_data(name='academy:register')
st.dataframe(redis_face_db)


#Real time Prediction