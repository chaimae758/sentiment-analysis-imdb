import streamlit as st
import joblib
import re

@st.cache_resource
def charger_modele():
    modele = joblib.load("modele_sentiment.pkl")
    vectorizer = joblib.load("vectorizer_tfidf.pkl")
    return modele, vectorizer

modele, vectorizer = charger_modele()

def nettoyer_texte(texte):
    texte = re.sub(r'<[^>]+>', ' ', texte)
    texte = texte.lower()
    texte = re.sub(r'[^a-z\s]', '', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

st.set_page_config(page_title="Analyse des Sentiments", page_icon="🎬")

st.title("🎬 Analyse des Sentiments — IMDb")
st.markdown("Écris un avis de film en anglais et découvre s'il est **positif** ou **négatif** !")
st.divider()

avis = st.text_area("✍️ Ton avis ici :", placeholder="Example: This movie was absolutely amazing...", height=150)

if st.button("🔍 Analyser le sentiment", type="primary"):
    if avis.strip() == "":
        st.warning("⚠️ Écris un avis avant d'analyser !")
    else:
        avis_nettoye = nettoyer_texte(avis)
        avis_vectorise = vectorizer.transform([avis_nettoye])
        prediction = modele.predict(avis_vectorise)[0]
        probabilite = modele.predict_proba(avis_vectorise)[0]
        confiance_pos = probabilite[1] * 100
        confiance_neg = probabilite[0] * 100

        st.divider()

        if prediction == "positive":
            st.success(f"## 😊 POSITIF — Confiance : {confiance_pos:.1f}%")
        else:
            st.error(f"## 😡 NÉGATIF — Confiance : {confiance_neg:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("😊 Positif")
            st.progress(int(confiance_pos))
            st.markdown(f"**{confiance_pos:.1f}%**")
        with col2:
            st.markdown("😡 Négatif")
            st.progress(int(confiance_neg))
            st.markdown(f"**{confiance_neg:.1f}%**")

st.divider()
st.caption("Modèle : Régression Logistique + TF-IDF | Accuracy : 89.83%")