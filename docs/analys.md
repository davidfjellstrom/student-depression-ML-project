# Analysdokumentation

Metod, resultat och slutsatser för projektet. Koden och alla diagram finns i notebooksen
`01_Dataforberedelse.ipynb` till `04_Supervised.ipynb`.

---

## Frågeställning

**Vilka livsstils- och studiefaktorer hänger starkast ihop med depression hos studenter –
och hur långt räcker de för att förutsäga den?**

En delfråga per notebook, bestämda innan analysen började:

- **`02_EDA`** – vilka variabler hänger ihop med `Depression`, och hur starkt?
- **`03_Unsupervised`** – går studenterna att dela in i naturliga grupper efter livsstil, och
  hänger grupperna i så fall ihop med om de är deprimerade?
- **`04_Supervised`** – räcker enkla modeller, eller behövs mer avancerade?

---

## Data och urval

[Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
(Kaggle, hopesb): **27 901 studenter**, 18 kolumner, binär målvariabel `Depression`.
Variabler som används:

| Variabel | Typ |
|---|---|
| Gender | binär |
| Age | numerisk |
| Academic Pressure | skala 0–5 |
| CGPA | numerisk (betyg) |
| Study Satisfaction | skala 0–5 |
| Sleep Duration | ordinal, 4 nivåer |
| Dietary Habits | ordinal, 3 nivåer |
| Work/Study Hours | numerisk |
| Financial Stress | skala 0–5 |
| Family History of Mental Illness | binär |

Målvariabeln är ganska jämnt fördelad – **58.6 % med depression, 41.4 % utan**. Ingen av
grupperna är alltså mycket mindre än den andra, så någon teknik som SMOTE behövs inte.

**Varför frågan om självmordstankar är utesluten.** `Have you ever had suicidal thoughts ?`
har starkast samband med målvariabeln (korrelation **0.55**), men frågan är ett symptom på
depression, inte något som kommer före den. En modell som använder den svarar i stort sett på
samma fråga som redan ställts, och blir därför inte användbar för att upptäcka någon tidigt.

Det förklarar varför resultatet här landar på 80 % accuracy, medan andra publika notebooks på
samma dataset ofta visar över 90 %. Siffran gäller vad som går att säga utan symptomfrågan.
Som jämförelse: eftersom 58.6 % av studenterna är deprimerade får man 58.6 % rätt bara genom
att alltid gissa "deprimerad". 80 % är alltså ungefär 21 procentenheter bättre än så.

---

## 1. Dataförberedelse

`01_Dataforberedelse.ipynb`

- Tog bort `id`, `Profession`, `Work Pressure`, `Job Satisfaction` och `Degree`. De var tomma
  eller irrelevanta, eftersom nästan alla i datasetet är heltidsstudenter.
- Tog bort rader där `Financial Stress` saknades, och rader med värdet `"Others"` i
  `Sleep Duration` och `Dietary Habits`.
- Rensade felinmatade värden i `City`.
- Kodade om sömn, kost och studienöjdhet till siffror i rätt ordning, och kön och
  familjehistorik till 0/1.
- Kontrollerade med en `assert` att inga saknade värden fanns kvar.

`City` togs inte med i modellerna. Skillnaderna mellan städerna var mindre än för
`Sleep Duration`, samtidigt som one-hot-kodning hade lagt till ett trettiotal kolumner.

Efter städningen finns **27 842 rader och 13 kolumner** kvar, sparade i
`student_depression_clean.csv`. Resten av projektet bygger på den filen.

---

## 2. Utforskande analys

`02_EDA.ipynb`

| Variabel | Korrelation med `Depression` |
|---|---:|
| Suicidal thoughts *(utesluten)* | 0.55 |
| Academic Pressure | 0.47 |
| Financial Stress | 0.36 |
| Age | -0.23 |
| Work/Study Hours | 0.21 |
| Dietary Habits | -0.21 |
| Study Satisfaction | -0.17 |
| Sleep Duration | -0.09 |
| Family History | 0.05 |
| CGPA | 0.02 |
| Gender | 0.00 |

Akademisk press och ekonomisk stress har klart starkast samband med depression, och
boxplottarna visar tydlig skillnad mellan de två grupperna. Ålder är den starkaste variabeln
som inte handlar om stress: andelen med depression sjunker jämnt från **71.3 % bland
18–21-åringar till 41.0 % bland 31–35-åringar**. Eftersom nedgången är så pass jämn togs
`Age` med i modellerna, trots att det inte är en livsstilsvariabel. Studienöjdhet följer samma
mönster men svagare (70.7 % → 47.2 %).

Sömn ser däremot svag ut (-0.09), och mönstret är ojämnt – 7–8 timmar ligger sämre till än
5–6 timmar. **Att sömn ser svag ut här men ändå visar en tydlig effekt när stressnivån hålls
konstant (avsnitt 3) är ett av de mest lärorika resultaten i projektet.**

---

## 3. Unsupervised learning

`03_Unsupervised.ipynb`

Klustringen gjordes på fem livsstilsvariabler: akademisk press, ekonomisk stress, sömn, kost
och arbets-/studietimmar. Algoritmen fick **aldrig se `Depression`**. Målvariabeln togs fram
först efteråt, för att se vad klustren motsvarade.

**Val av antal kluster.** Måtten tyder på att det inte finns några tydliga kluster:
elbow-kurvan har ingen knyck, silhouette ligger på 0.16–0.18 för alla K mellan 2 och 10, och
när klustringen körs om med olika slumpfrön blir indelningen bara stabil för K=2. Det sista
mäts med ARI, som visar hur lika två indelningar är: 0.998 för K=2, men bara 0.737 för K=4 och
0.426 för K=10. PCA pekar åt samma håll – de två viktigaste komponenterna fångar bara
**45.8 %** av variationen i datan.

**K=4 valdes ändå**, men som ett sätt att få fram en hypotes, inte som ett färdigt resultat.
K=2 delar bara upp studenterna efter stressnivå, vilket EDA:n redan visat. K=4 delar dessutom
högstress-gruppen efter sömn, och det var ett mönster som ingen enskild variabel hade visat.

| Kluster | n | Press | Ek. stress | Sömn | Timmar | Depression |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 197 | 3.93 | 3.93 | 1.34 (<5h) | 8.54 | **85.5 %** |
| 2 | 7 454 | 3.73 | 3.77 | 3.49 (7–8h) | 8.59 | **77.5 %** |
| 3 | 6 561 | 2.63 | 2.71 | 2.49 | 2.06 | 34.7 % |
| 0 | 6 630 | 2.14 | 2.01 | 2.23 | 9.10 | 31.6 % |

Kluster 1 och 2 har hög stress, 0 och 3 låg. Trots att algoritmen aldrig såg målvariabeln
skiljer det nästan **54 procentenheter** i depressionsandel mellan det högsta och det lägsta
klustret – mer än vad någon enskild variabel gav i EDA:n.

**Hypotesen om sömn, testad separat.** Kluster 1 och 2 har nästan samma stressnivå men olika
sömn, och skiljer sig 8 procentenheter. Det gav en hypotes som kom ur analysen i stället för i
förväg: *sömn spelar roll även när stressen är lika hög.* Eftersom klusterindelningen inte var
stabil testades mönstret direkt mot datan i stället, genom att dela upp studenterna efter
stressnivå och titta på sömn inom varje nivå:

| Sömn | Hög stress | Låg stress |
|---|---:|---:|
| <5h | 87.7 % | 31.0 % |
| 5–6h | 81.5 % | 25.9 % |
| 7–8h | 83.6 % | 26.4 % |
| >8h | 78.2 % | 19.7 % |

Hypotesen stämde, men inte riktigt som förväntat. Skillnaden mellan minst och mest sömn är
ungefär lika stor i båda grupperna (~10 procentenheter). **Sömn dämpar alltså inte hög stress
särskilt – kopplingen finns oavsett stressnivå.**

---

## 4. Supervised learning

`04_Supervised.ipynb`

**Upplägg: testsetet hålls undan.** Datan delades 80/20 innan modelleringen började: 22 273
studenter att träna på och **5 569 undanlagda**. Delningen är stratifierad, alltså med samma
andel deprimerade i båda delarna. Modellerna jämfördes sedan
med 5-fold cross-validation på enbart träningsdatan. `StandardScaler` ligger
inuti varje `Pipeline`, så att skalningen räknas om för varje fold och inte påverkas av
valideringsdelen. Modellen valdes utifrån CV-resultaten, **innan** testsetet användes.

Anledningen är att om man väljer modell efter att ha sett testresultaten, blir testsetet en
del av modellvalet. Då säger siffran inte längre hur bra modellen är på ny data.

**Modelltävlingen.** Tre modeller som gör olika antaganden om datan:

| Modell | CV AUC-ROC (träningsdata) |
|---|---:|
| Logistisk regression | 0.8709 ± 0.0070 |
| Gradient boosting (HistGradientBoosting) | 0.8694 ± 0.0078 |
| Random Forest (200 träd) | 0.8590 ± 0.0075 |

Modellerna ligger nära varandra. Skillnaden mellan de två bästa är 0.0015, vilket är mycket
mindre än spridningen mellan foldarna (±0.007) – den skillnaden går alltså inte att lita på.
Det enda som är någorlunda säkert är att Random Forest ligger lite efter.

**Logistisk regression valdes.** Den fick högst CV-AUC, och två saker till talade för den:
resultatet går att läsa av koefficient för koefficient, vilket känns viktigt när ämnet är
studenters psykiska hälsa, och modellen är enkel – inga hyperparametrar att ställa in, samma
resultat varje körning, snabb att träna om.

**Feature importance ska tolkas försiktigt.** Random Forest rankar `CGPA` som näst viktigast,
trots att korrelationen med `Depression` bara är 0.02. En trolig förklaring är att `CGPA` har
hundratals olika värden och alltså många ställen att dela datan på, medan skalvariablerna bara
har 3–5. Måttet verkar alltså gynna variabler med många möjliga delningar, så listan säger mer
om variablernas form än om hur viktiga de är.

---

## 5. Deep learning

Ett feed-forward-nätverk i TensorFlow/Keras med 10 features in, två dolda lager (64 och 32
neuroner, ReLU) och ett utlager med sigmoid. Binary crossentropy, Adam och batch size 256.
**Early stopping** följde valideringsförlusten och tog tillbaka de bästa vikterna.
Valideringsdatan togs från träningsdatan (15 %), så testsetet användes inte här heller. Nätet
tränade 23 epoker, med bästa vikter från epok 13, och nådde validerings-AUC 0.8756 utan tecken
på överanpassning.

Nätet kördes inte med 5-fold cross-validation. Dess siffra bygger alltså på en enda uppdelning
i stället för fem, och går inte att jämföra rakt av med CV-siffrorna ovan.

Testresultatet blev **accuracy 80.1 %, AUC 0.8742**, alltså i nivå med logistisk regression
och inte bättre. Det är ett väntat resultat på tabelldata av den här storleken – neurala nät
är framför allt starka på data som bilder, text och ljud. **Deep learning är alltså inte
automatiskt bättre än klassisk maskininlärning.**

---

## Modelljämförelse

| Modell | Accuracy | Precision | Recall | F1 | AUC-ROC (test) | AUC-ROC (träning) |
|---|---:|---:|---:|---:|---:|---:|
| Chansa på majoriteten (baslinje) | 58.6 % | – | – | – | 0.500 | – |
| Random Forest | 78.6 % | 0.803 | 0.842 | 0.822 | 0.8616 | 0.8590 ± 0.0075 |
| Gradient boosting | 79.7 % | 0.812 | 0.851 | 0.831 | 0.8737 | 0.8694 ± 0.0078 |
| Neuralt nätverk (MLP) | 80.1 % | 0.825 | 0.837 | 0.831 | 0.8742 | 0.8756 *(val-split)* |
| **Logistisk regression** *(vald)* | **80.0 %** | 0.816 | 0.851 | 0.833 | **0.8749** | 0.8709 ± 0.0070 |

Baslinjen är inte tränad. Den motsvarar att alltid gissa den vanligaste klassen.

**CV-valet stämde med testresultaten.** Modellerna hamnade i samma ordning på testsetet som i
cross-validationen, och test-AUC ligger för alla något *över* CV-AUC. Ingen av modellerna är
alltså överanpassad. Alla fyra hamnar mellan AUC 0.86 och 0.87, vilket tyder på att **det är
informationen i datat som sätter gränsen, inte valet av modell.**

---

## Slutsatser och begränsningar

- Akademisk press och ekonomisk stress hänger starkast ihop med depression. Ålder är den näst
  starkaste faktorn. CGPA och kön har nästan inget samband alls.
- Livsstilsvariablerna bildar **inga tydliga grupper**. Studenterna ligger snarare på en
  glidande skala, där stressnivån är det som skiljer dem åt mest. Att klustringen inte hittade
  några grupper var i sig ett svar på delfrågan.
- Klustringen gav ändå en användbar hypotes om sömn. Den stämde när den testades för sig, men
  effekten var ungefär lika stor oavsett stressnivå (~10 procentenheter), inte extra skyddande
  vid hög stress.
- **Enkla modeller räckte.** Logistisk regression, gradient boosting och det neurala nätet gav
  i stort sett samma resultat.
- Korrelationsanalys, klustring och feature importance ger samma bild av datat. Att tre olika
  metoder pekar åt samma håll gör bilden mer trovärdig.

Fyra saker att ha i åtanke:

- **Samband, inte orsak.** Datat är en ögonblicksbild utan tidsdimension. Att dålig sömn
  hänger ihop med depression säger inget om vilken av dem som orsakar den andra.
- **Självrapporterad data.** Både målvariabeln och de flesta features bygger på studenternas
  egna svar. Resultaten gäller dessutom bara den här studentgruppen och har inte testats på
  någon annan.

---

## Vidareutveckling

- **Slå ihop press och ekonomisk stress till en variabel.** Klustringen antydde att de
  tillsammans skiljer grupper åt bättre än var för sig, och det går att testa direkt.
- **Justera beslutströskeln och kontrollera sannolikheterna.** PR-kurvan visar att det går att
  byta precision mot recall. Sannolikheterna som appen visar borde också jämföras med hur det
  faktiskt blev för grupper med liknande svar.
- **Data över tid.** Det är det enda som skulle göra det möjligt att säga något om orsak och
  riktning.
