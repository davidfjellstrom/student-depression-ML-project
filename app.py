"""
Streamlit-app kring champion-modellen från 04_Supervised.ipynb.

Appen visar hur variablerna TILLSAMMANS påverkar sannolikheten för depression.
Åtta färdiga profiler (A-H) spänner upp de tre faktorer projektet identifierat
som viktigast - ålder, stressnivå och sömn - så att man kan jämföra två profiler
som skiljer sig på exakt en punkt och se den enskilda faktorns bidrag.

Kör med:  ./venv/bin/streamlit run app.py
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Sidinställningar och stil
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Depression bland studenter", page_icon="📊",
                   layout="wide")

# Samma palett som i notebookarna, så appen ser ut som resten av projektet.
PALETTE = {
    "blue": "#2a78d6",
    "red": "#e34948",
    "grid": "#e1e0d9",
    "ink": "#0b0b0b",
    "muted": "#898781",
    "surface": "#fcfcfb",
}

DATASET_SNITT = 0.586  # andel med depression i hela datasetet
VALJ = "— Välj en profil —"


# ---------------------------------------------------------------------------
# Ladda modell och data. @st.cache_* gör att filerna bara läses EN gång, inte
# vid varje reglagedrag.
# ---------------------------------------------------------------------------
@st.cache_resource
def ladda_modell():
    bundle = joblib.load("champion_model.joblib")
    return bundle["pipeline"], bundle["features"]


@st.cache_data
def ladda_faktiska_andelar():
    """Den faktiska depressionsandelen i varje profils motsvarande grupp.

    Grupperna definieras av samma tre axlar som profilerna: ålder (ung 18-22 /
    äldre 28+), stressnivå (Academic Pressure + Financial Stress, låg <=5 /
    hög >=7) och sömn (lite <7h / mycket >=7h). Räknas ut ur datasetet vid
    körning, så att siffrorna aldrig kan bli inaktuella.
    """
    df = pd.read_csv("student_depression_clean.csv")
    stress = df["Academic Pressure"] + df["Financial Stress"]
    df = df.assign(
        _alder=pd.cut(df["Age"], [17, 22, 27, 60], labels=["ung", "mellan", "äldre"]),
        _stress=pd.cut(stress, [-1, 5, 6, 10], labels=["låg", "mellan", "hög"]),
        _somn=df["Sleep Duration"].map({1: "lite", 2: "lite", 3: "mycket", 4: "mycket"}),
    )
    df = df[df["_alder"].isin(["ung", "äldre"]) & df["_stress"].isin(["låg", "hög"])]
    grupper = df.groupby(["_alder", "_stress", "_somn"], observed=True)["Depression"]
    return {nyckel: (andel, antal) for nyckel, andel, antal
            in zip(grupper.mean().index, grupper.mean().values, grupper.size().values)}


pipeline, feature_cols = ladda_modell()
faktiska = ladda_faktiska_andelar()


# ---------------------------------------------------------------------------
# De åtta profilerna
#
# Uppbyggnaden är en 2x2x2: ålder x stress x sömn. De sju övriga variablerna är
# LÅSTA vid datasetets medianvärden i alla åtta profiler - annars hade skillnaden
# mellan två profiler innehållit mer än den faktor man tror sig jämföra.
# Därför går det att isolera en enskild faktor genom att jämföra rätt par:
#   A vs B -> sömn      A vs C -> ålder      A vs E -> stress
# ---------------------------------------------------------------------------
LASTA = {
    "kon": "Man",          # datasetets median; kön saknar mätbar effekt (korr. 0.00)
    "cgpa": 7.8,           # median 7.77
    "noejdhet": 3,
    "kost": "Måttlig",
    "timmar": 8,
    "familj": "Nej",
}

ALDER = {"ung": 20, "äldre": 31}
STRESS = {"hög": 4, "låg": 2}          # sätts på både press och ekonomi
SOMN = {"lite": "<5h", "mycket": "7-8h"}

PROFILER = {}
for bokstav, (a, s, so) in zip("ABCDEFGH", [
    ("ung", "hög", "lite"), ("ung", "hög", "mycket"),
    ("äldre", "hög", "lite"), ("äldre", "hög", "mycket"),
    ("ung", "låg", "lite"), ("ung", "låg", "mycket"),
    ("äldre", "låg", "lite"), ("äldre", "låg", "mycket"),
]):
    beskrivning = (
        f"{'ung' if a == 'ung' else 'äldre'}, "
        f"{'hög' if s == 'hög' else 'låg'} press och stress, "
        f"{'lite' if so == 'lite' else 'god'} sömn"
    )
    PROFILER[bokstav] = {
        "beskrivning": beskrivning,
        "axlar": (a, s, so),
        "varden": {**LASTA, "alder": ALDER[a], "press": STRESS[s],
                   "ekonomi": STRESS[s], "somn": SOMN[so]},
    }

ETIKETTER = {f"Profil {b} — {p['beskrivning']}": b for b, p in PROFILER.items()}


# ---------------------------------------------------------------------------
# Sidopanel: profilval överst, därefter reglagen
# ---------------------------------------------------------------------------
st.sidebar.header("Studentprofil")

vald_etikett = st.sidebar.selectbox(
    "Färdig profil", [VALJ] + list(ETIKETTER),
    help="Fyller i alla reglagen på en gång. Justera dem sedan fritt.",
)

# När användaren byter profil skrivs reglagens värden om. Detta måste ske INNAN
# reglagen skapas nedan - då fungerar session_state som deras startvärde.
if vald_etikett != st.session_state.get("_forra_profil"):
    st.session_state["_forra_profil"] = vald_etikett
    if vald_etikett != VALJ:
        for nyckel, varde in PROFILER[ETIKETTER[vald_etikett]]["varden"].items():
            st.session_state[nyckel] = varde
        st.session_state["_har_valt"] = True

st.sidebar.divider()
st.sidebar.caption("Eller ställ in värdena själv:")

alder = st.sidebar.slider("Ålder", 18, 59, 24, key="alder")
kon = st.sidebar.radio("Kön", ["Kvinna", "Man"], horizontal=True, key="kon")

st.sidebar.markdown("**Press och stress**")
press = st.sidebar.slider("Akademisk press", 1, 5, 3, key="press",
                          help="1 = mycket låg, 5 = mycket hög")
ekonomi = st.sidebar.slider("Ekonomisk stress", 1, 5, 3, key="ekonomi",
                            help="1 = mycket låg, 5 = mycket hög")
timmar = st.sidebar.slider("Arbets-/studietimmar per dag", 0, 12, 7, key="timmar")

st.sidebar.markdown("**Livsstil**")
somn = st.sidebar.select_slider("Sömn per natt", options=["<5h", "5-6h", "7-8h", ">8h"],
                                value="7-8h", key="somn")
kost = st.sidebar.select_slider("Kostvanor", options=["Ohälsosam", "Måttlig", "Hälsosam"],
                                value="Måttlig", key="kost")

st.sidebar.markdown("**Studier och bakgrund**")
noejdhet = st.sidebar.slider("Studienöjdhet", 1, 5, 3, key="noejdhet",
                             help="1 = mycket missnöjd, 5 = mycket nöjd")
cgpa = st.sidebar.slider("CGPA (betygssnitt)", 0.0, 10.0, 7.7, step=0.1, key="cgpa")
familj = st.sidebar.radio("Psykisk ohälsa i familjen", ["Nej", "Ja"], horizontal=True,
                          key="familj")

if st.sidebar.button("Nollställ", width="stretch"):
    for nyckel in ("_har_valt", "_forra_profil"):
        st.session_state.pop(nyckel, None)
    st.rerun()

# Reglagen räknas som "rörda" så fort de avviker från startläget. Tillsammans med
# profilvalet avgör det om vi har något att visa - appen ska inte påstå någon
# sannolikhet innan användaren gjort ett aktivt val.
STARTLAGE = {"alder": 24, "kon": "Kvinna", "press": 3, "ekonomi": 3, "timmar": 7,
             "somn": "7-8h", "kost": "Måttlig", "noejdhet": 3, "cgpa": 7.7,
             "familj": "Nej"}
if any(st.session_state.get(k) != v for k, v in STARTLAGE.items()):
    st.session_state["_har_valt"] = True
har_valt = st.session_state.get("_har_valt", False)


# ---------------------------------------------------------------------------
# Huvudsida
# ---------------------------------------------------------------------------
st.title("Depression bland studenter")
st.markdown(
    "Verktyget bygger på en **logistisk regression** tränad på 27 842 studenter "
    "(Student Depression Dataset, Kaggle). Modellen valdes i "
    "`04_Supervised.ipynb` och når **test-AUC 0.875**, **80% träffsäkerhet** och "
    "**85% recall** för depressionsklassen."
)

if not har_valt:
    # ----- Startläget: ingen siffra, bara en inbjudan och profilöversikten -----
    st.info(
        "**Välj en färdig profil i sidopanelen** — eller ställ in värdena själv. "
        "Ingen skattning visas innan du gjort ett val."
    )
    st.subheader("De åtta profilerna")
    st.markdown(
        "Profilerna är byggda som en systematisk 2×2×2 över de tre faktorer "
        "projektet identifierat som viktigast: **ålder**, **press och stress** och "
        "**sömn**. Övriga sju variabler är låsta vid datasetets medianvärden i "
        "samtliga profiler — därför beror varje skillnad mellan två profiler enbart "
        "på de tre faktorerna, och man kan isolera en i taget:"
    )
    st.markdown(
        "- **A mot B** — allt lika utom sömnen\n"
        "- **A mot C** — allt lika utom åldern\n"
        "- **A mot E** — allt lika utom press och stress"
    )

    oversikt = pd.DataFrame([
        {
            "Profil": b,
            "Ålder": p["axlar"][0],
            "Press och stress": p["axlar"][1],
            "Sömn": p["axlar"][2],
            "Faktisk andel i gruppen": f"{faktiska[p['axlar']][0]*100:.1f} %",
            "Antal studenter": f"{faktiska[p['axlar']][1]:,}".replace(",", " "),
        }
        for b, p in PROFILER.items()
    ])
    st.dataframe(oversikt, hide_index=True, width="stretch")
    st.caption(
        "Andelarna är uträknade direkt ur datasetet och beskriver de verkliga "
        "grupperna — inte modellens gissning. De fungerar därför som "
        "rimlighetskontroll när du väljer en profil."
    )

else:
    # ----- Något är valt: visa skattningen -----
    indata = pd.DataFrame([{
        "Gender": 1 if kon == "Man" else 0,
        "Age": float(alder),
        "Academic Pressure": float(press),
        "CGPA": float(cgpa),
        "Study Satisfaction": float(noejdhet),
        "Sleep Duration": {"<5h": 1, "5-6h": 2, "7-8h": 3, ">8h": 4}[somn],
        "Dietary Habits": {"Ohälsosam": 1, "Måttlig": 2, "Hälsosam": 3}[kost],
        "Work/Study Hours": float(timmar),
        "Financial Stress": float(ekonomi),
        "Family History of Mental Illness": 1 if familj == "Ja" else 0,
    }])[feature_cols]

    sannolikhet = float(pipeline.predict_proba(indata)[0, 1])

    vanster, hoger = st.columns([1, 1.4])

    with vanster:
        if vald_etikett != VALJ:
            st.subheader(f"Profil {ETIKETTER[vald_etikett]}")
            st.caption(PROFILER[ETIKETTER[vald_etikett]]["beskrivning"].capitalize())
        else:
            st.subheader("Egen profil")
            st.caption("Värdena är inställda för hand.")

        st.markdown(
            f"<div style='font-size:4.5rem; font-weight:700; line-height:1.1; "
            f"color:{PALETTE['blue'] if sannolikhet < 0.5 else PALETTE['red']}'>"
            f"{sannolikhet*100:.1f}%</div>",
            unsafe_allow_html=True,
        )
        st.progress(sannolikhet)
        st.caption("Modellens skattade sannolikhet för depression.")

        # Vid en oförändrad färdig profil kan vi jämföra mot verkligheten.
        if vald_etikett != VALJ:
            axlar = PROFILER[ETIKETTER[vald_etikett]]["axlar"]
            andel, antal = faktiska[axlar]
            orord = all(st.session_state.get(k) == v for k, v
                        in PROFILER[ETIKETTER[vald_etikett]]["varden"].items())
            if orord:
                st.metric("Faktisk andel i motsvarande grupp",
                          f"{andel*100:.1f} %",
                          delta=f"{(sannolikhet - andel)*100:+.1f} p.e. mot modellen",
                          delta_color="off")
                st.caption(f"Baserat på {antal:,} verkliga studenter i datasetet."
                           .replace(",", " "))
            else:
                st.caption("Du har ändrat i profilen — jämförelsen mot den verkliga "
                           "gruppen visas bara för en orörd profil.")

        differens = (sannolikhet - DATASET_SNITT) * 100
        st.caption(
            f"Datasetets snitt är {DATASET_SNITT*100:.1f} %. Den här profilen ligger "
            f"{abs(differens):.1f} procentenheter "
            f"{'över' if differens > 0 else 'under'} snittet."
        )

        st.markdown("**Klassificering**")
        st.write(f"Vid tröskel 0.50: **{'Depression' if sannolikhet >= 0.5 else 'Ingen depression'}**")
        st.write(f"Vid tröskel 0.35 (screening): **{'Depression' if sannolikhet >= 0.35 else 'Ingen depression'}**")
        st.caption(
            "En lägre tröskel fångar fler verkliga fall till priset av fler falska "
            "larm — rimligt när kostnaden för att missa någon är hög."
        )

    # -----------------------------------------------------------------------
    # Vad driver skattningen? Logistisk regression är tolkningsbar: varje
    # variabels bidrag är dess koefficient gånger det standardiserade värdet.
    # -----------------------------------------------------------------------
    with hoger:
        st.subheader("Vad driver skattningen?")

        scaler = pipeline.named_steps["scaler"]
        modell = pipeline.named_steps["model"]
        skalad = scaler.transform(indata)[0]
        bidrag = pd.Series(modell.coef_[0] * skalad, index=feature_cols)

        visningsnamn = {
            "Gender": "Kön",
            "Age": "Ålder",
            "Academic Pressure": "Akademisk press",
            "CGPA": "CGPA",
            "Study Satisfaction": "Studienöjdhet",
            "Sleep Duration": "Sömn",
            "Dietary Habits": "Kostvanor",
            "Work/Study Hours": "Studietimmar",
            "Financial Stress": "Ekonomisk stress",
            "Family History of Mental Illness": "Familjehistorik",
        }
        bidrag = bidrag.rename(visningsnamn).sort_values()

        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor(PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])
        farger = [PALETTE["red"] if v > 0 else PALETTE["blue"] for v in bidrag.values]
        ax.barh(bidrag.index, bidrag.values, color=farger)
        ax.axvline(0, color=PALETTE["muted"], linewidth=1)
        ax.set_xlabel("Bidrag till skattningen")
        ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Röda staplar drar upp sannolikheten, blå drar ned den. Längden visar hur "
            "mycket — beräknat som modellens koefficient gånger det standardiserade "
            "värdet för just den här profilen."
        )

# ---------------------------------------------------------------------------
# Ansvarsfriskrivning - viktig, och sist så att den syns efter resultatet.
# ---------------------------------------------------------------------------
st.divider()
st.warning(
    "**Detta är ett skolprojekt, inte ett medicinskt verktyg.** Modellen är tränad "
    "på enkätsvar från ett tvärsnitt utan tidsdimension — den visar samband, inte "
    "orsak. Symptomvariabler (t.ex. självmordstankar) är medvetet uteslutna, "
    "eftersom de ingår i själva diagnosen. Siffran ovan är en statistisk skattning "
    "för en grupp med liknande svar, inte en bedömning av en enskild person. "
    "Mår du eller någon i din närhet dåligt — kontakta vården."
)
