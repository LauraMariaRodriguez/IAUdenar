import joblib
import math
import pandas as pd

dt = joblib.load("dt.pkl")  #Cargamos el arbol de decision
rf = joblib.load("rf.pkl")  #Cargamos el random forest
lr  = joblib.load("lr.pkl")  #Cargamos la linear regresion
ab = joblib.load("ab.pkl")  #Cargamos Adaboost 

meanVal = joblib.load("meanVal.pkl")  #Cargamos los valores medios
meanVal = pd.DataFrame(meanVal)
maxVal = joblib.load("maxVal.pkl")  #Cargamos los valores máximos
maxVal = pd.DataFrame(maxVal)
minVal = joblib.load("minVal.pkl")  #Cargamos los valores mínimos
minVal = pd.DataFrame(minVal)
corr = joblib.load("correlations.pkl")
corr = dict(zip(meanVal.columns, corr))
val = joblib.load("meanVal.pkl") 
val = pd.DataFrame(val)


import streamlit as st 

def rr_to_hb(rr):
	rr = 1/rr
	rr = rr*1000*60
	return rr

st.set_page_config(layout="wide")
st.title('IA Udenar')
st.header("Ejercicio IA Udenar Oscar Andrés Rosero Calderón")
st.subheader("Maestría en Electrónica")

st.write("Stress Wearables")

left, right = st.beta_columns(2)

hrv_MEAN_RR = right.slider("Latidos por minuto", math.floor(rr_to_hb(minVal.hrv_MEAN_RR)), math.floor(rr_to_hb (maxVal.hrv_MEAN_RR)) + 1, step = 1)
hrv_MEAN_RR = 1/(hrv_MEAN_RR/1000/60)

right.markdown(
	"<center><img src ='https://raw.githubusercontent.com/orosero/IAUdenar/main/apple-watch.png' style = 'width : 100%;'><br> Image source: <a href = 'https://www.freepng.es/png-01knxm/'>elijahkey122</a></center>"	
	, unsafe_allow_html=True)

sliders = []
def addSli(var, text, place = None):

	minim = float(minVal[var])
	maxim = float(maxVal[var])

	inc = 0
	while maxim - minim < 0.1:
		maxim = maxim*10
		minim = minim*10
		inc = inc+1
	if inc > 0:
		text = text+" · 10^"+str(inc)

	if place :
		sliders.append([
			var,
			place.slider(text, minim, maxim, step = (maxim-minim)/10 )
			])

	else:
		sliders.append([
			var,
			st.slider(text, minim, maxim, step = (maxim-minim)/10 )
			])

addSli("eda_MEAN", "Actividad electrodermica media", left)
left.markdown(
        "<center><img src ='https://raw.githubusercontent.com/orosero/IAUdenar/main/bracelet-sensor.png' style = 'width : 70%;'><br> Image source: <a href = 'https://www.freepng.es/png-js7wxa/'>rolandosumme716</a> </center>"
	, unsafe_allow_html=True)


sc = ["hrv_MEAN_RR", "eda_MEAN", "baseline", "meditation", "stress", "amusement", "hrv_KURT_SQUARE", "eda_MEAN_2ND_GRAD_CUBE"]   #special cases

state = left.selectbox("Situación actual",("Normal","Emocionado", "Estresado", "Meditando"))

with st.beta_expander("Configuración avanzada"):
	col1, col2, col3 = st.beta_columns(3)
	num = len(val.columns)//3

	for i in val.columns[:num]:
		if i not in sc:
			addSli(i,i,col1)

	for i in val.columns[num : 2*(num+1)]:
		if i not in sc:
			addSli(i,i,col2)

	for i in val.columns[2*(num+1) :]:
		if i not in sc:
			addSli(i,i,col3)
def update():

	val.hrv_MEAN_RR = hrv_MEAN_RR

	for i in sliders:
		val[i[0]] = i[1]


	val.hrv_KURT_SQUARE = val.hrv_KURT**2
	val.eda_MEAN_2ND_GRAD_CUBE = val.eda_MEAN_2ND_GRAD ** 3

	val.baseline = 1 if state == "Normal" else 0
	val.amusement = 1 if state == "Emocionado" else 0
	val.stress = 1 if state == "Estresado" else 0
	val.meditation = 1 if state == "Meditando" else 0

modelo = left.selectbox("Modelo de predicción",("Árbol de Decisión","Random Forest", "Linear Regression", "AdaBosst"))

if modelo == 'Árbol de Decisión':
	st.text('Árbol de Decisión')
	nStress = int(dt.predict(val))
elif modelo == 'Random Forest':
	st.text('Random Forest')
	nStress = int(rf.predict(val))
elif modelo == 'Linear Regression':
	st.text('Linear Regression')
	nStress = int(lr.predict(val))
elif modelo == 'AdaBosst':
	st.text('AdaBosst')
	nStress = int(ab.predict(val))
else:
	st.text('error')	

if st.button('Predict'):
			update()
			
			prediction = dt.predict(val)
		
	
			st.write('Results 🔍')
		
			st.text(nStress)
			if nStress < 3:
				st.text("Que estres ni estres, si te relajas más te quedas dormido")
			elif nStress <5:
				st.text("Nivel de estres normal")
			else:
				st.text("Nivel de estres alto, deberias relajarte")
