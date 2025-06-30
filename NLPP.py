#core pkgs

import streamlit as st
import altair as alt

#eda pkgs
import pandas as pd 
import numpy as np

#utils
import joblib


pipe_lr = joblib.load(open(r"models/emotion_classifier_pipe_lr_03_june_2024.pkl", "rb"))
#fxn

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠" , "disgust": "🤢","fear":"😱😨" ,"happy":"😃😊", "joy":"☺️😅☺️", "neutral":"😐" , "sadness":"😟", "surprise":"😲","shame":"🙂‍↔️"}


def main():
    st.title("Emotion Classifier App")
    menu = ["home" , "monitor ", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice  == "home":
        st.subheader("Home-Emotion in Text")

        with st.form(key='emotion_clf_Text'):
            raw_text = st.text_area("type Here")
            submit_text = st.form_submit_button(label='Submit')
        
        if submit_text:
            col1, col2 = st.columns(2)

            #apply fxn here 
            prediction = predict_emotion(raw_text)
            Probability = get_prediction_proba(raw_text)

            with col1:
                st.success("original Text")
                st.write(raw_text)

                st.success("prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(Probability)))


            with col2:
                st.success("prediction Probability")
                #st.write(Probability)
                proba_df = pd.DataFrame(Probability,columns= pipe_lr.classes_)
                #st.write(proba_df.T)
                Proba_df_clean= proba_df.T.reset_index()
                Proba_df_clean.columns = ["Emotions","probability"]

                fig = alt.Chart(Proba_df_clean).mark_bar().encode(x='Emotions',y ="probability" , color = 'Emotions')
                st.altair_chart(fig, use_container_width= True)


        

    elif choice == "monitor":
        st.subheader("monitor App")

    else:
        st.subheader("about")

if __name__ == '__main__':
    main()
