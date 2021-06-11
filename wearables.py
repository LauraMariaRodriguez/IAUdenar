rf = joblib.load("models/rf.pkl")  #Cargamos el Gradiente Boost
dt = joblib.load("models/dt.pkl")  #Cargamos el Gradiente Boost
lr = joblib.load("models/lr.pkl")  #Cargamos el Gradiente Boost
GBC = joblib.load("models/GBC.pkl")  #Cargamos el Gradiente Boost

meanVal = joblib.load("data/meanVal.pkl")  #Cargamos los valores medios
meanVal = pd.DataFrame(meanVal)
maxVal = joblib.load("data/maxVal.pkl")  #Cargamos los valores medios
maxVal = pd.DataFrame(maxVal)
minVal = joblib.load("data/minVal.pkl")  #Cargamos los valores medios
minVal = pd.DataFrame(minVal)
corr = joblib.load("data/correlations.pkl")
corr = dict(zip(meanVal.columns, corr))
val = joblib.load("data/meanVal.pkl") 
val = pd.DataFrame(val)


import streamlit as st



def stress_o_meter2(level):
	return '''<style type="text/css">
			#arrow{
				position: relative;
				height: 30vh;
				width: 10vw;
				top: -6em;
			}
			#level{
				height: 5em;
				width: 15em;
			}
		</style>
		<center>
		<div>
			<img src="https://i.ibb.co/nCKKMH9/sem.png" id="level">
			<br>
			<img src="https://drive.google.com/uc?export=view&id=1XmIuIxpmMRjw3Xf6e9AqO7pgrMx274w_" id="arrow">
		</div>
		</center>

		<script type="text/javascript" src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4="crossorigin="anonymous"></script>

		<script type="text/javascript">
			$(document).ready(function(){
				$( "#arrow" ).animate({
	    			left: "'''+str( 2.4*(level -3) +0.2 )+'''em",
	  			}, 1500)
			})
		</script>'''


st.set_page_config(layout="wide")

st.markdown(
	'''
	<style>
		div[data-testid='stHorizontalBlock'] div[role='slider']{
			background-color : black
		}
	</style>''',unsafe_allow_html = True)





st.title("Taller IA: Predictor de estrés")
st.header("Laura Rodríguez y Harold Ruiz")


def rr_to_hb(rr):
	rr = 1/rr
	rr = rr*1000*60
	return rr

left, mid, right = st.beta_columns((4,1,4))

# ATENCION el maximo y el minimo se invierten al pasar de RR a BPM
maxim = math.floor(rr_to_hb(minVal.hrv_MEAN_RR))
minim = math.floor(rr_to_hb(maxVal.hrv_MEAN_RR))

hrv_MEAN_RR = right.slider("Latidos por minuto", minim, maxim, step = 1, value = (minim + math.floor((maxim-minim)/2))   )
hrv_MEAN_RR = 1/(hrv_MEAN_RR/1000/60)


right.markdown(
	'''<center>
		<img src ='data:image/jpeg;base64,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' style = 'width : 32%;'> 
		<br> Image by: <a href = 'https://www.amazon.com/-/es/inteligente-SmartWatch-compatible-rastreador-inteligentes/dp/B08F54L2FP'>Google Images</a>
	</center''', 
	unsafe_allow_html=True)

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
			place.slider(text, minim, maxim, step = (maxim-minim)/10, value = (maxim-minim)/2 + minim )
			])

	else:
		sliders.append([
			var,
			st.slider(text, minim, maxim, step = (maxim-minim)/10, value = (maxim-minim)/2 + minim )
			])

addSli("eda_MEAN", "Actividad electrodermica media", left)


sc = ["hrv_MEAN_RR", "eda_MEAN", "baseline", "meditation", "stress", "amusement"]   #special cases

center = st.beta_columns((1,2,1))
state = left.selectbox("Situación actual",("Normal","Emocionado", "Estresado", "Meditando"))

with st.beta_expander("Configuración avanzada	(Permite acceder a todas las variables del modelo)"):
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


	val.baseline = 1 if state == "Normal" else 0
	val.amusement = 1 if state == "Emocionado" else 0
	val.stress = 1 if state == "Estresado" else 0
	val.meditation = 1 if state == "Meditando" else 0
	
			
modelo = left.selectbox("Modelo de predicción",("Random Forest","Gradiente Boost", "Regresion Lineal","Árbol de Decisión"))


if   modelo == 'Random Forest':
	st.text('Random Forest')
	nStress = int(rf.predict(val))
elif modelo == 'Gradiente Boost':
	st.text('Grandient Boost')
	nStress = int(GBC.predict(val))
elif modelo == 'Regresion Lineal':
	st.text('Regresion Lineal')
	nStress = int(lr.predict(val))
elif modelo == 'Árbol de Decisión':
	st.text('Árbol de Decisión')
	nStress = int(dt.predict(val))
else:
	st.text('error')	

if st.button('Predecir'):
			update()
			
			st.write('''
			## Resultado 🔍 
			''')
			if nStress < 3:
				st.text("Estres bajo")
			elif nStress <5:
				st.text("Nivel de estres normal")
			else:
				st.text("Nivel de estres alto, deberias relajarte")
			st.components.v1.html(stress_o_meter2(nStress))
			
