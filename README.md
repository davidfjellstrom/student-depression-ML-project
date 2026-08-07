# Går det att förutsäga depression hos studenter?

**Vilka livsstils- och studiefaktorer hänger starkast ihop med depression hos studenter –
och hur långt räcker de för att förutsäga den?**

Ett komplett ML-projekt från rådata till utvärderad modell: dataförberedelse, EDA,
unsupervised learning för hypotesgenerering, och fyra klassificeringsmodeller inklusive
ett neuralt nätverk – samt en interaktiv Streamlit-app kring den modell som valdes.

## Avgränsningar

**Symptomvariabler utesluts.** `Have you ever had suicidal thoughts ?` har den starkaste
kopplingen till målvariabeln i hela datasetet (korrelation 0.55), men utesluts medvetet:
suicidtankar är enligt DSM-5 ett *symptom* på depression, inte en oberoende riskfaktor.
En modell som förutsäger en diagnos från dess eget symptom mäter mest sin egen läcka. Det
är skälet till att projektet landar på 80% accuracy där publika notebooks på samma dataset
ofta visar 90%+ – siffran här beskriver vad som går att säga *innan* symptomen tillfrågas,
vilket också är den enda siffran som är meningsfull i en screening-kontext.

**Samband, inte orsak.** Datat är ett tvärsnitt utan tidsdimension. Att dålig sömn hänger
ihop med depression säger ingenting om vilken riktning pilen går.

## Data

Student Depression Dataset (Kaggle, hopesb). 27 901 rader och 18 kolumner i rådata;
27 842 rader och 13 kolumner efter städning. Målvariabeln `Depression` är binär och
rimligt balanserad (58.6% / 41.4%), så ingen balanseringsteknik som SMOTE behövs.

## Struktur

| Notebook | Delfråga | Innehåll |
|---|---|---|
| `01_Dataforberedelse.ipynb` | – | Städning, saknade värden, ordinal- och binärkodning |
| `02_EDA.ipynb` | Vilka variabler samvarierar med Depression, och hur starkt? | Distributioner, korrelationer, visuell verifiering |
| `03_Unsupervised.ipynb` | Bildar livsstilsvariablerna naturliga grupper? | KMeans, PCA, stabilitetstest, hypotesgenerering |
| `04_Supervised.ipynb` | Räcker enkla modeller, eller krävs flexibla och djupa? | Fyra modeller, cross-validation, slututvärdering |

Kör i ordning 01 → 04. `01` producerar `student_depression_clean.csv` som resten bygger på.

Utöver notebooksen finns `export_model.py`, som sparar champion-modellen till
`champion_model.joblib`, och `app.py` – Streamlit-appen som beskrivs nedan.

## Interaktiv app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Appen låter en ställa in en studentprofil och se vad champion-modellen skattar för
sannolikhet, tillsammans med en nedbrytning av vilka variabler som driver skattningen –
möjlig just för att den valda modellen är tolkningsbar.

**Åtta färdiga profiler (A–H)** visar hur variablerna samverkar. De är byggda som en
systematisk 2×2×2 över de tre faktorer projektet identifierat som viktigast: ålder,
press/stress och sömn. Övriga sju variabler är låsta vid datasetets medianvärden i
samtliga profiler, så att skillnaden mellan två profiler enbart beror på de tre
faktorerna – då går det att isolera en i taget (A mot B ger sömnens bidrag, A mot C
ålderns, A mot E stressens). De faktiska depressionsandelarna i motsvarande grupper
spänner från 15.9% till 90.6% och visas bredvid modellens skattning som
rimlighetskontroll.

Ingen skattning visas innan användaren gjort ett aktivt val – appen påstår inget om en
profil som inte efterfrågats. `export_model.py` tränar om modellen med samma features,
split och seed som `04_Supervised.ipynb` och verifierar mot notebookens test-AUC 0.8749,
så att notebooken själv kan lämnas orörd.

Appen är ett skolprojekt och inte ett medicinskt verktyg – den visar samband, inte orsak,
och skattar för grupper med liknande svar, inte för enskilda personer.

## Svar på frågeställningen

**Vilka faktorer?** Akademisk press (0.47) och ekonomisk stress (0.36) dominerar. Ålder är
den starkaste icke-stressvariabeln (-0.23): andelen med depression faller från 71.3% bland
18–21-åringar till 41.0% bland 31–35-åringar. Därefter arbets-/studietimmar (0.21),
kostvanor (-0.21) och studienöjdhet (-0.17). CGPA (0.02) och kön (0.00) saknar i praktiken
betydelse. Tre oberoende metoder pekar åt samma håll: korrelationsanalysen i EDA:n,
klustringen i steg 3 och feature importance i steg 4.

**Hur långt räcker de?** Champion-modellen – logistisk regression – når test-AUC 0.8749,
accuracy 80.0% och recall 85.1% för depressionsklassen vid tröskel 0.5. Hela fältet ryms
inom AUC 0.86–0.87: gradient boosting 0.8737, neuralt nät 0.8742, Random Forest 0.8616.
Taket sätts av informationen i datat, inte av modellvalet.

## Vad projektet visade på vägen

**Klustringen hittade inga grupper – och det var svaret.** Silhouette 0.16–0.18 genom hela
K-intervallet, K=4 bara måttligt stabil (ARI ≈ 0.74). Studenterna ligger längs ett
kontinuum där stressnivå dominerar, inte i distinkta livsstilstyper. Klustren används
därför som hypotesgenerator, inte som segment.

**En hypotes ur klustringen, prövad separat.** Två kluster med nästan identisk stressnivå
men olika sömn skilde sig 8 procentenheter i depressionsandel. Hypotesen prövades direkt
mot datan utan KMeans – och höll, men med en korrigering: sömneffekten är ungefär lika stor
(~10 procentenheter) oavsett stressnivå. Sömn är alltså inte en skyddsfaktor som specifikt
dämpar hög stress.

**Enkelt slog komplext.** Logistisk regression är i praktiken likvärdig med gradient
boosting och det neurala nätet, och valdes för sin tolkningsbarhet. Att djupinlärning inte
vann är väntat på strukturerad tabelldata av den här storleken.

**Feature importance ska läsas kritiskt.** Random Forest rankar CGPA tvåa (0.171) trots
korrelation 0.02 i EDA:n – impurity-baserad importance överskattar systematiskt
kontinuerliga variabler med många unika värden.

## Metod

Stratifierad 80/20-split; testsetet låses in innan modellering. `StandardScaler` inuti
varje `Pipeline` så att skalningen görs om per fold. Modellval på 5-fold stratifierad
cross-validation med AUC-ROC, innan testsetet rörs. Utvärdering med Confusion Matrix,
Classification Report, AUC-ROC och precision-recall. Neuralt nät: Keras-MLP 64 → 32 → 1,
ReLU/sigmoid, binary crossentropy, Adam, early stopping.

## Frågeställning kontra hypotes

Frågeställningen och delfrågorna ställdes i förväg och styr projektets upplägg.
Sömn-hypotesen i `03_Unsupervised.ipynb` gjorde det inte – den uppstod ur klustringen och
prövades i efterhand. De två hålls medvetet isär.
