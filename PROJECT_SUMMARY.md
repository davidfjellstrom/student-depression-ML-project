# Stort Individuellt Projekt – Sammanfattning & arbetsgång

## Syfte
Visa att du behärskar **hela ML-processen** från rådata till utvärderad modell. Projektet är tänkt att kunna visas upp på jobbintervjuer.

## Hårda krav (får inte missas)
- Måste inkludera **unsupervised learning** – för att förstå datan eller generera hypoteser (klustring; PCA/UMAP vid behov av dimensionsreduktion).
- Måste testa **flera algoritmer** inom supervised learning, **inklusive deep learning** (strukturerad eller ostrukturerad data – bild, text, ljud).
- ⚠️ **FÅR INTE** enbart använda linjär eller multipel linjär regression som modell.
- Utvärdera med metrik som passar problemtypen: **Confusion Matrix, Classification Report, AUC-ROC, precision-recall**.

## Projektbeslut

- **Frågeställning:** Vilka livsstils- och studiefaktorer hänger starkast ihop med depression hos studenter – och hur långt räcker de för att förutsäga den?
  - **Delfråga 02 (EDA):** Vilka variabler samvarierar med `Depression`, och hur starkt?
  - **Delfråga 03 (Unsupervised):** Bildar livsstilsvariablerna naturliga grupper – och hänger de i så fall ihop med utfallet?
  - **Delfråga 04 (Supervised):** Räcker enkla modeller för att förutsäga `Depression`, eller krävs flexibla och djupa?
- **Dataset:** Student Depression Dataset (Kaggle, hopesb) – 27 901 studenter med livsstils- och studierelaterade variabler
- **Målvariabel:** `Depression` (binär: ja/nej)
- **Nyckelvariabler:** Academic Pressure, CGPA, Study Satisfaction, Sleep Duration, Dietary Habits, Work/Study Hours, Financial Stress, Family History of Mental Illness, Suicidal thoughts, Gender, Age, City
- **Kolumner att granska kritiskt:** Profession, Work Pressure, Job Satisfaction, Degree, id – troligen mest tomma/irrelevanta då de flesta i datasetet är heltidsstudenter
- **Känslig variabel:** "Have you ever had suicidal thoughts?" – hantera varsamt, håll analys på gruppnivå
- **Installerade paket:** pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter
- **Nästa steg:** Skapa `01_EDA.ipynb`, ladda in datan, undersöka saknade värden/extremvärden, utforska hypoteser kring vilka variabler som hänger ihop med Depression

## Arbetsgång

**Bärande princip: arbeta baklänges från kraven.** Utvärderingsmetriken ovan är klassificeringsmetrik, så projektet behöver ett **klassificeringsproblem** någonstans – och målvariabeln `Depression` ger det direkt.

1. **Dataförberedelse.** Hantera saknade värden, koda om kategoriska variabler (Gender, City, Dietary Habits, Sleep Duration etc.), städa bort/utvärdera irrelevanta kolumner, skapa ett rent, analysklart dataset.
2. **EDA.** Utforska datan visuellt och statistiskt – testa hypoteser kring sömn, akademisk press, ekonomisk stress och familjehistorik mot Depression. Undersök korstabeller, boxplottar och korrelationer. Håll koll på balansen i målvariabeln.
3. **Unsupervised learning.** Klustring på livsstilsvariablerna (utan att titta på Depression-etiketten) för att se om naturliga grupper bildas – och hur de överlappar med den faktiska diagnosen. PCA/UMAP vid behov av dimensionsreduktion.
4. **Supervised learning + deep learning.** Träna flera algoritmer (t.ex. logistisk regression, Random Forest, SVM) och minst ett neuralt nätverk. Jämför prestanda och tolkningsbarhet.
5. **Utvärdering.** Confusion Matrix, Classification Report, AUC-ROC och precision-recall – passar direkt eftersom målvariabeln är binär.
6. **Extra (frivilligt, ger guldstjärnor).** Interaktiva visualiseringar/dashboards eller deployment med Streamlit, Dash, Flask eller React.js.

## Praktisk uppstart
- Projektmapp: `student-depression-ML-project`
- Virtuell miljö (`venv`) skapad och aktiverad
- Paket installerade, `requirements.txt` sparad
- Dataset nedladdat som zip från Kaggle och uppackat i projektmappen

## Övrigt
Jag kommer göra samtliga pushar till github, du kan däremot få förse mig med ihoppaketerade git commit meddelanden (där du i punktform sammanfattar vad som gjorts sedan senaste git pushen) + git push meddelande, så att jag mer eller mindre bara kan klicka avfyra.

---
*Uppdaterad version av projektinstruktionerna, anpassad för Student Depression Dataset. Ursprungligen destillerad från presentationen "Large Individual Project" (Vladimir JN Bykov).*
