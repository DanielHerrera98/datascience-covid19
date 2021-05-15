import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


#Título
st.header("Análise de vulnerabilidade COVID-19 no estado de São Paulo")
st.markdown("Este algoritmo foi construído utilizando o modelo de Floresta aleatória (Random forest) ")


# Coleta dos dados
df = pd.read_csv('dfclean.csv')

#Cabeçalho
st.subheader('Informações dos dados')

#Nome do usuário
user_input = st.sidebar.text_input('Digite seu nome')





# Limpeza dos dados

df['idade'] = df['idade'].fillna(0)
df['idade'] = df['idade'].astype(int)
df['obito'] = df['obito'].fillna(0)
df['obito'] = df['obito'].astype(int)
df['cs_sexo'] = df['cs_sexo'].replace('MASCULINO','1')
df['cs_sexo'] = df['cs_sexo'].replace('FEMININO','2')
df['cs_sexo'] = df['cs_sexo'].replace('INDEFINIDO','0')
df['cs_sexo'] = df['cs_sexo'].replace('IGNORADO','0')
df['cs_sexo'] = df['cs_sexo'].fillna(0)
df['cs_sexo'] = df['cs_sexo'].astype(int)


df = df.loc[ 
   (df['asma'] != 'IGNORADO') &
   (df['cardiopatia'] != 'IGNORADO') &
   (df['diabetes'] != 'IGNORADO') &
   (df['doenca_hematologica'] != 'IGNORADO') &
   (df['doenca_hepatica'] != 'IGNORADO') &
   (df['doenca_neurologica'] != 'IGNORADO') &
   (df['doenca_renal'] != 'IGNORADO') &
   (df['imunodepressao'] != 'IGNORADO') &
   (df['obesidade'] != 'IGNORADO') &
   (df['outros_fatores_de_risco'] != 'IGNORADO') &
   (df['pneumopatia'] != 'IGNORADO') &
   (df['puerpera'] != 'IGNORADO') &
   (df['sindrome_de_down'] != 'IGNORADO')]

df = df.dropna()

doencas = ['asma','cardiopatia','diabetes','doenca_hematologica','doenca_hepatica','doenca_neurologica','doenca_renal','imunodepressao','obesidade','outros_fatores_de_risco','pneumopatia','puerpera','sindrome_de_down']

df[doencas] = df[doencas].replace('NÃO','0')
df[doencas] = df[doencas].replace('SIM','1')
	
df[doencas] = df[doencas].astype(int)

df['codigo_ibge'] = df['codigo_ibge'].fillna(0)
df['codigo_ibge'] = df['codigo_ibge'].astype(int)

#Análise exploratória dos dados

def EDA():
	plt.subplots(figsize=(11, 8))
	sns.heatmap(df.corr(),  annot=True, annot_kws={"size": 10})
	
	

# Modelagem

X = df[['codigo_ibge','idade','cs_sexo','asma','cardiopatia','diabetes','doenca_hematologica','doenca_hepatica','doenca_neurologica','doenca_renal','imunodepressao','obesidade','outros_fatores_de_risco','pneumopatia','puerpera','sindrome_de_down']]
Y = df['obito']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


def get_user_data():
	nome_munic = st.sidebar.selectbox("Cidade", (
		'São Paulo','Adamantina','Adolfo','Aguaí','Águas da Prata','Águas de Lindóia','Águas de São Pedro','Agudos','Alfredo Marcondes','Altair','Altinópolis','Alto Alegre','Alumínio','Álvares Florence','Álvares Machado','Álvaro de Carvalho','Alvinlândia','Americana','Américo Brasiliense','Américo de Campos','Amparo','Analândia','Andradina','Angatuba','Anhembi','Anhumas','Aparecida','Aparecida dOeste','Apiaí','Araçariguama','Araçatuba','Araçoiaba da Serra','Aramina','Arandu','Arapeí','Araraquara','Araras','Arco-Íris','Arealva','Areias','Areiópolis','Ariranha','Artur Nogueira','Arujá','Aspásia','Assis','Atibaia','Auriflama','Avaí','Avanhandava','Avaré','Bady Bassitt','Balbinos','Bálsamo','Bananal','Barão de Antonina','Barbosa','Bariri','Barra Bonita','Barra do Chapéu','Barra do Turvo','Barretos','Barrinha','Barueri','Bastos','Batatais','Bauru','Bebedouro','Bento de Abreu','Bernardino de Campos','Bertioga','Bilac','Birigui','Biritiba Mirim','Boa Esperança do Sul','Bocaina','Bofete','Boituva','Bom Jesus dos Perdões','Bom Sucesso de Itararé','Borá','Boracéia','Borborema','Borebi','Botucatu','Bragança Paulista','Braúna','Brejo Alegre','Brodowski','Brotas','Buri','Buritama','Buritizal','Cabrália Paulista','Cabreúva','Caçapava','Cachoeira Paulista','Caconde','Cafelândia','Caiabu','Caieiras','Caiuá','Cajamar','Cajati','Cajobi','Cajuru','Campina do Monte Alegre','Campinas','Campo Limpo Paulista','Campos do Jordão','Campos Novos Paulista','Cananéia','Canas','Cândido Mota','Cândido Rodrigues','Canitar','Capão Bonito','Capela do Alto','Capivari','Caraguatatuba','Carapicuíba','Cardoso','Casa Branca','Cássia dos Coqueiros','Castilho','Catanduva','Catiguá','Cedral','Cerqueira César','Cerquilho','Cesário Lange','Charqueada','Chavantes','Clementina','Colina','Colômbia','Conchal','Conchas','Cordeirópolis','Coroados','Coronel Macedo','Corumbataí','Cosmópolis','Cosmorama','Cotia','Cravinhos','Cristais Paulista','Cruzália','Cruzeiro','Cubatão','Cunha','Descalvado','Diadema','Dirce Reis','Divinolândia','Dobrada','Dois Córregos','Dolcinópolis','Dourado','Dracena','Duartina','Dumont','Echaporã','Eldorado','Elias Fausto','Elisiário','Embaúba','Embu das Artes','Embu-Guaçu','Emilianópolis','Engenheiro Coelho','Espírito Santo do Pinhal','Espírito Santo do Turvo','Estiva Gerbi','Estrela do Norte','Estrela dOeste','Euclides da Cunha Paulista','Fartura','Fernando Prestes','Fernandópolis','Fernão','Ferraz de Vasconcelos','Flora Rica','Floreal','Flórida Paulista','Florínea','Franca','Francisco Morato','Franco da Rocha','Gabriel Monteiro','Gália','Garça','Gastão Vidigal','Gavião Peixoto','General Salgado','Getulina','Glicério','Guaiçara','Guaimbê','Guaíra','Guapiaçu','Guapiara','Guará','Guaraçaí','Guaraci','Guarani dOeste','Guarantã','Guararapes','Guararema','Guaratinguetá','Guareí','Guariba','Guarujá','Guarulhos','Guatapará','Guzolândia','Herculândia','Holambra','Hortolândia','Iacanga','Iacri','Iaras','Ibaté','Ibirá','Ibirarema','Ibitinga','Ibiúna','Icém','Iepê','Igaraçu do Tietê','Igarapava','Igaratá','Iguape','Ilha Comprida','Ilha Solteira','Ilhabela','Indaiatuba','Indiana','Indiaporã','Inúbia Paulista','Ipaussu','Iperó','Ipeúna','Ipiguá','Iporanga','Ipuã','Iracemápolis','Irapuã','Irapuru','Itaberá','Itaí','Itajobi','Itaju','Itanhaém','Itaoca','Itapecerica da Serra','Itapetininga','Itapeva','Itapevi','Itapira','Itapirapuã Paulista','Itápolis','Itaporanga','Itapuí','Itapura','Itaquaquecetuba','Itararé','Itariri','Itatiba','Itatinga','Itirapina','Itirapuã','Itobi','Itu','Itupeva','Ituverava','Jaborandi','Jaboticabal','Jacareí','Jaci','Jacupiranga','Jaguariúna','Jales','Jambeiro','Jandira','Jardinópolis','Jarinu','Jaú','Jeriquara','Joanópolis','João Ramalho','José Bonifácio','Júlio Mesquita','Jumirim','Jundiaí','Junqueirópolis','Juquiá','Juquitiba','Laranjal Paulista','Lavínia','Lavrinhas','Leme','Lençóis Paulista','Limeira','Lindóia','Lins','Lorena','Lourdes','Louveira','Lucélia','Luís Antônio','Luiziânia','Lupércio','Lutécia','Macatuba','Macaubal','Macedônia','Magda','Mairinque','Mairiporã','Manduri','Marabá Paulista','Maracaí','Marapoama','Mariápolis','Marília','Marinópolis','Martinópolis','Matão','Mauá','Mendonça','Meridiano','Mesópolis','Miguelópolis','Mineiros do Tietê','Mira Estrela','Miracatu','Mirandópolis','Mirante do Paranapanema','Mirassol','Mirassolândia','Mococa','Mogi das Cruzes','Mogi Guaçu','Mogi Mirim','Mombuca','Monções','Mongaguá','Monte Alegre do Sul','Monte Alto','Monte Aprazível','Monte Azul Paulista','Monte Castelo','Monte Mor','Morro Agudo','Morungaba','Motuca','Murutinga do Sul','Nantes','Narandiba','Natividade da Serra','Nazaré Paulista','Neves Paulista','Nhandeara','Nipoã','Nova Aliança','Nova Campina','Nova Canaã Paulista','Nova Europa','Nova Granada','Nova Guataporanga','Nova Independência','Nova Luzitânia','Nova Odessa','Novais','Novo Horizonte','Nuporanga','Ocauçu','Olímpia','Onda Verde','Oriente','Orindiúva','Orlândia','Osasco','Oscar Bressane','Osvaldo Cruz','Ourinhos','Ouro Verde','Ouroeste','Pacaembu','Palestina','Palmares Paulista','Palmeira dOeste','Palmital','Panorama','Paraguaçu Paulista','Paraibuna','Paraíso','Paranapanema','Paranapuã','Parapuã','Pardinho','Pariquera-Açu','Parisi','Patrocínio Paulista','Paulicéia','Paulínia','Paulistânia','Paulo de Faria','Pederneiras','Pedra Bela','Pedranópolis','Pedregulho','Pedreira','Pedrinhas Paulista','Pedro de Toledo','Penápolis','Pereira Barreto','Pereiras','Peruíbe','Piacatu','Piedade','Pilar do Sul','Pindamonhangaba','Pindorama','Pinhalzinho','Piquerobi','Piquete','Piracaia','Piracicaba','Piraju','Pirajuí','Pirangi','Pirapora do Bom Jesus','Pirapozinho','Pirassununga','Piratininga','Pitangueiras','Planalto','Platina','Poá','Poloni','Pompéia','Pongaí','Pontal','Pontalinda','Pontes Gestal','Populina','Porangaba','Porto Feliz','Porto Ferreira','Potim','Potirendaba','Pracinha','Pradópolis','Praia Grande','Pratânia','Presidente Alves','Presidente Bernardes','Presidente Epitácio','Presidente Prudente','Presidente Venceslau','Promissão','Quadra','Quatá','Queiroz','Queluz','Quintana','Rafard','Rancharia','Redenção da Serra','Regente Feijó','Reginópolis','Registro','Restinga','Ribeira','Ribeirão Bonito','Ribeirão Branco','Ribeirão Corrente','Ribeirão do Sul','Ribeirão dos Índios','Ribeirão Grande','Ribeirão Pires','Ribeirão Preto','Rifaina','Rincão','Rinópolis','Rio Claro','Rio das Pedras','Rio Grande da Serra','Riolândia','Riversul','Rosana','Roseira','Rubiácea','Rubinéia','Sabino','Sagres','Sales','Sales Oliveira','Salesópolis','Salmourão','Saltinho','Salto','Salto de Pirapora','Salto Grande','Sandovalina','Santa Adélia','Santa Albertina','Santa Bárbara dOeste','Santa Branca','Santa Clara dOeste','Santa Cruz da Conceição','Santa Cruz da Esperança','Santa Cruz das Palmeiras','Santa Cruz do Rio Pardo','Santa Ernestina','Santa Fé do Sul','Santa Gertrudes','Santa Isabel','Santa Lúcia','Santa Maria da Serra','Santa Mercedes','Santa Rita do Passa Quatro','Santa Rita dOeste','Santa Rosa de Viterbo','Santa Salete','Santana da Ponte Pensa','Santana de Parnaíba','Santo Anastácio','Santo André','Santo Antônio da Alegria','Santo Antônio de Posse','Santo Antônio do Aracanguá','Santo Antônio do Jardim','Santo Antônio do Pinhal','Santo Expedito','Santópolis do Aguapeí','Santos','São Bento do Sapucaí','São Bernardo do Campo','São Caetano do Sul','São Carlos','São Francisco','São João da Boa Vista','São João das Duas Pontes','São João de Iracema','São João do Pau dAlho','São Joaquim da Barra','São José da Bela Vista','São José do Barreiro','São José do Rio Pardo','São José do Rio Preto','São José dos Campos','São Lourenço da Serra','São Luiz do Paraitinga','São Manuel','São Miguel Arcanjo','São Pedro','São Pedro do Turvo','São Roque','São Sebastião','São Sebastião da Grama','São Simão','São Vicente','Sarapuí','Sarutaiá','Sebastianópolis do Sul','Serra Azul','Serra Negra','Serrana','Sertãozinho','Sete Barras','Severínia','Silveiras','Socorro','Sorocaba','Sud Mennucci','Sumaré','Suzanápolis','Suzano','Tabapuã','Tabatinga','Taboão da Serra','Taciba','Taguaí','Taiaçu','Taiúva','Tambaú','Tanabi','Tapiraí','Tapiratiba','Taquaral','Taquaritinga','Taquarituba','Taquarivaí','Tarabai','Tarumã','Tatuí','Taubaté','Tejupá','Teodoro Sampaio','Terra Roxa','Tietê','Timburi','Torre de Pedra','Torrinha','Trabiju','Tremembé','Três Fronteiras','Tuiuti','Tupã','Tupi Paulista','Turiúba','Turmalina','Ubarana','Ubatuba','Ubirajara','Uchoa','União Paulista','Urânia','Uru','Urupês','Valentim Gentil','Valinhos','Valparaíso','Vargem','Vargem Grande do Sul','Vargem Grande Paulista','Várzea Paulista','Vera Cruz','Vinhedo','Viradouro','Vista Alegre do Alto','Vitória Brasil','Votorantim','Votuporanga','Zacarias',

		))
	#cod_munic = st.sidebar.selectbox("Código do município", ('1234','5678'))
	#mes_sintomas = st.sidebar.slider("Mês do início dos sintomas", 1,12,1)
	idade = st.sidebar.slider("Idade", 0,130,22)
	sexo = st.sidebar.selectbox("Sexo", ('Masculino','Feminino'))
	_asma = st.sidebar.selectbox("Possui asma?", ('Não','Sim'))
	_cardiopatia = st.sidebar.selectbox("Possui problemas de coração?", ('Não','Sim'))
	_diabetes = st.sidebar.selectbox("Possui diabetes?", ('Não','Sim'))
	_obesidade = st.sidebar.selectbox("Possui obesidade?", ('Não','Sim'))
	_pneumopatia = st.sidebar.selectbox("Possui problemas no pulmão?", ('Não','Sim'))
	_doenca_hematologica = st.sidebar.selectbox("Possui doença hematológica?", ('Não','Sim'))
	_doenca_hepatica = st.sidebar.selectbox("Possui doença hepática?", ('Não','Sim'))
	_doenca_neurologica = st.sidebar.selectbox("Possui doença neurológica?", ('Não','Sim'))
	_doenca_renal = st.sidebar.selectbox("Possui doença renal?", ('Não','Sim'))
	_imunodepressao = st.sidebar.selectbox("Possui imunodepressão?", ('Não','Sim'))
	_outros_fatores_de_risco = st.sidebar.selectbox("Possui outros fatores de risco?", ('Não','Sim'))
	_puerpera = st.sidebar.selectbox("Gerou algum filho a pouco tempo?", ('Não','Sim'))
	_sindrome_de_down = st.sidebar.selectbox("Possui sindrome de down?", ('Não','Sim'))

	if _asma == 'Não':
		asma = 0
	if _asma == 'Sim':
		asma = 1

	if _cardiopatia == 'Não':
		cardiopatia = 0
	if _cardiopatia == 'Sim':
		cardiopatia = 1

	if _diabetes == 'Não':
		diabetes = 0
	if _diabetes == 'Sim':
		diabetes = 1

	if _doenca_hematologica == 'Não':
		doenca_hematologica = 0
	if _doenca_hematologica == 'Sim':
		doenca_hematologica = 1

	if _doenca_hepatica == 'Não':
		doenca_hepatica = 0
	if _doenca_hepatica == 'Sim':
		doenca_hepatica = 1

	if _doenca_neurologica == 'Não':
		doenca_neurologica = 0
	if _doenca_neurologica == 'Sim':
		doenca_neurologica = 1

	if _doenca_renal == 'Não':
		doenca_renal = 0
	if _doenca_renal == 'Sim':
		doenca_renal = 1

	if _imunodepressao == 'Não':
		imunodepressao = 0
	if _imunodepressao == 'Sim':
		imunodepressao = 1

	if _obesidade == 'Não':
		obesidade = 0
	if _obesidade == 'Sim':
		obesidade = 1

	if _outros_fatores_de_risco == 'Não':
		outros_fatores_de_risco = 0
	if _outros_fatores_de_risco == 'Sim':
		outros_fatores_de_risco = 1

	if _pneumopatia == 'Não':
		pneumopatia = 0
	if _pneumopatia == 'Sim':
		pneumopatia = 1

	if _puerpera == 'Não':
		puerpera = 0
	if _puerpera == 'Sim':
		puerpera = 1

	if _sindrome_de_down == 'Não':
		sindrome_de_down = 0
	if _sindrome_de_down == 'Sim':
		sindrome_de_down = 1




	if sexo == 'Masculino':
		cs_sexo = 1
	if sexo == 'Feminino':
		cs_sexo = 2

	if nome_munic == 'Adamantina': 
		cod_munic = 3500105
	if nome_munic == 'Adolfo': 
		cod_munic = 3500204
	if nome_munic == 'Aguaí': 
		cod_munic = 3500303
	if nome_munic == 'Águas da Prata': 
		cod_munic = 3500402
	if nome_munic == 'Águas de Lindóia': 
		cod_munic = 3500501
	if nome_munic == 'Águas de São Pedro': 
		cod_munic = 3500600
	if nome_munic == 'Agudos': 
		cod_munic = 3500709
	if nome_munic == 'Alfredo Marcondes': 
		cod_munic = 3500808
	if nome_munic == 'Altair': 
		cod_munic = 3500907
	if nome_munic == 'Altinópolis': 
		cod_munic = 3501004
	if nome_munic == 'Alto Alegre': 
		cod_munic = 3501103
	if nome_munic == 'Alumínio': 
		cod_munic = 3501152
	if nome_munic == 'Álvares Florence': 
		cod_munic = 3501202
	if nome_munic == 'Álvares Machado': 
		cod_munic = 3501301
	if nome_munic == 'Álvaro de Carvalho': 
		cod_munic = 3501400
	if nome_munic == 'Alvinlândia': 
		cod_munic = 3501509
	if nome_munic == 'Americana': 
		cod_munic = 3501608
	if nome_munic == 'Américo Brasiliense': 
		cod_munic = 3501707
	if nome_munic == 'Américo de Campos': 
		cod_munic = 3501806
	if nome_munic == 'Amparo': 
		cod_munic = 3501905
	if nome_munic == 'Analândia': 
		cod_munic = 3502002
	if nome_munic == 'Andradina': 
		cod_munic = 3502101
	if nome_munic == 'Angatuba': 
		cod_munic = 3502200
	if nome_munic == 'Anhembi': 
		cod_munic = 3502309
	if nome_munic == 'Anhumas': 
		cod_munic = 3502408
	if nome_munic == 'Aparecida': 
		cod_munic = 3502507
	if nome_munic == 'Aparecida dOeste': 
		cod_munic = 3502606
	if nome_munic == 'Apiaí': 
		cod_munic = 3502705
	if nome_munic == 'Araçariguama': 
		cod_munic = 3502754
	if nome_munic == 'Araçatuba': 
		cod_munic = 3502804
	if nome_munic == 'Araçoiaba da Serra': 
		cod_munic = 3502903
	if nome_munic == 'Aramina': 
		cod_munic = 3503000
	if nome_munic == 'Arandu': 
		cod_munic = 3503109
	if nome_munic == 'Arapeí': 
		cod_munic = 3503158
	if nome_munic == 'Araraquara': 
		cod_munic = 3503208
	if nome_munic == 'Araras': 
		cod_munic = 3503307
	if nome_munic == 'Arco-Íris': 
		cod_munic = 3503356
	if nome_munic == 'Arealva': 
		cod_munic = 3503406
	if nome_munic == 'Areias': 
		cod_munic = 3503505
	if nome_munic == 'Areiópolis': 
		cod_munic = 3503604
	if nome_munic == 'Ariranha': 
		cod_munic = 3503703
	if nome_munic == 'Artur Nogueira': 
		cod_munic = 3503802
	if nome_munic == 'Arujá': 
		cod_munic = 3503901
	if nome_munic == 'Aspásia': 
		cod_munic = 3503950
	if nome_munic == 'Assis': 
		cod_munic = 3504008
	if nome_munic == 'Atibaia': 
		cod_munic = 3504107
	if nome_munic == 'Aur	iflama': 
		cod_munic = 3504206
	if nome_munic == 'Avaí': 
		cod_munic = 3504305
	if nome_munic == 'Avanhandava': 
		cod_munic = 3504404
	if nome_munic == 'Avaré': 
		cod_munic = 3504503
	if nome_munic == 'Bady Bassitt': 
		cod_munic = 3504602
	if nome_munic == 'Balbinos': 
		cod_munic = 3504701
	if nome_munic == 'Bálsamo': 
		cod_munic = 3504800
	if nome_munic == 'Bananal': 
		cod_munic = 3504909
	if nome_munic == 'Barão de Antonina': 
		cod_munic = 3505005
	if nome_munic == 'Barbosa': 
		cod_munic = 3505104
	if nome_munic == 'Bariri': 
		cod_munic = 3505203
	if nome_munic == 'Barra Bonita': 
		cod_munic = 3505302
	if nome_munic == 'Barra do Chapéu': 
		cod_munic = 3505351
	if nome_munic == 'Barra do Turvo': 
		cod_munic = 3505401
	if nome_munic == 'Barretos': 
		cod_munic = 3505500
	if nome_munic == 'Barrinha': 
		cod_munic = 3505609
	if nome_munic == 'Barueri': 
		cod_munic = 3505708
	if nome_munic == 'Bastos': 
		cod_munic = 3505807
	if nome_munic == 'Batatais': 
		cod_munic = 3505906
	if nome_munic == 'Bauru': 
		cod_munic = 3506003
	if nome_munic == 'Bebedouro': 
		cod_munic = 3506102
	if nome_munic == 'Bento de Abreu': 
		cod_munic = 3506201
	if nome_munic == 'Bernardino de Campos': 
		cod_munic = 3506300
	if nome_munic == 'Bertioga': 
		cod_munic = 3506359
	if nome_munic == 'Bilac': 
		cod_munic = 3506409
	if nome_munic == 'Birigui': 
		cod_munic = 3506508
	if nome_munic == 'Biritiba Mirim': 
		cod_munic = 3506607
	if nome_munic == 'Boa Esperança do Sul': 
		cod_munic = 3506706
	if nome_munic == 'Bocaina': 
		cod_munic = 3506805
	if nome_munic == 'Bofete': 
		cod_munic = 3506904
	if nome_munic == 'Boituva': 
		cod_munic = 3507001
	if nome_munic == 'Bom Jesus dos Perdões': 
		cod_munic = 3507100
	if nome_munic == 'Bom Sucesso de Itararé': 
		cod_munic = 3507159
	if nome_munic == 'Borá': 
		cod_munic = 3507209
	if nome_munic == 'Boracéia': 
		cod_munic = 3507308
	if nome_munic == 'Borborema': 
		cod_munic = 3507407
	if nome_munic == 'Borebi': 
		cod_munic = 3507456
	if nome_munic == 'Botucatu': 
		cod_munic = 3507506
	if nome_munic == 'Bragança Paulista': 
		cod_munic = 3507605
	if nome_munic == 'Braúna': 
		cod_munic = 3507704
	if nome_munic == 'Brejo Alegre': 
		cod_munic = 3507753
	if nome_munic == 'Brodowski': 
		cod_munic = 3507803
	if nome_munic == 'Brotas': 
		cod_munic = 3507902
	if nome_munic == 'Buri': 
		cod_munic = 3508009
	if nome_munic == 'Buritama': 
		cod_munic = 3508108
	if nome_munic == 'Buritizal': 
		cod_munic = 3508207
	if nome_munic == 'Cabrália Paulista': 
		cod_munic = 3508306
	if nome_munic == 'Cabreúva': 
		cod_munic = 3508405
	if nome_munic == 'Caçapava': 
		cod_munic = 3508504
	if nome_munic == 'Cachoeira Paulista': 
		cod_munic = 3508603
	if nome_munic == 'Caconde': 
		cod_munic = 3508702
	if nome_munic == 'Cafelândia': 
		cod_munic = 3508801
	if nome_munic == 'Caiabu': 
		cod_munic = 3508900
	if nome_munic == 'Caieiras': 
		cod_munic = 3509007
	if nome_munic == 'Caiuá': 
		cod_munic = 3509106
	if nome_munic == 'Cajamar': 
		cod_munic = 3509205
	if nome_munic == 'Cajati': 
		cod_munic = 3509254
	if nome_munic == 'Cajobi': 
		cod_munic = 3509304
	if nome_munic == 'Cajuru': 
		cod_munic = 3509403
	if nome_munic == 'Campina do Monte Alegre': 
		cod_munic = 3509452
	if nome_munic == 'Campinas': 
		cod_munic = 3509502
	if nome_munic == 'Campo Limpo Paulista': 
		cod_munic = 3509601
	if nome_munic == 'Campos do Jordão': 
		cod_munic = 3509700
	if nome_munic == 'Campos Novos Paulista': 
		cod_munic = 3509809
	if nome_munic == 'Cananéia': 
		cod_munic = 3509908
	if nome_munic == 'Canas': 
		cod_munic = 3509957
	if nome_munic == 'Cândido Mota': 
		cod_munic = 3510005
	if nome_munic == 'Cândido Rodrigues': 
		cod_munic = 3510104
	if nome_munic == 'Canitar': 
		cod_munic = 3510153
	if nome_munic == 'Capão Bonito': 
		cod_munic = 3510203
	if nome_munic == 'Capela do Alto': 
		cod_munic = 3510302
	if nome_munic == 'Capivari': 
		cod_munic = 3510401
	if nome_munic == 'Caraguatatuba': 
		cod_munic = 3510500
	if nome_munic == 'Carapicuíba': 
		cod_munic = 3510609
	if nome_munic == 'Cardoso': 
		cod_munic = 3510708
	if nome_munic == 'Casa Branca': 
		cod_munic = 3510807
	if nome_munic == 'Cássia dos Coqueiros': 
		cod_munic = 3510906
	if nome_munic == 'Castilho': 
		cod_munic = 3511003
	if nome_munic == 'Catanduva': 
		cod_munic = 3511102
	if nome_munic == 'Catiguá': 
		cod_munic = 3511201
	if nome_munic == 'Cedral': 
		cod_munic = 3511300
	if nome_munic == 'Cerqueira César': 
		cod_munic = 3511409
	if nome_munic == 'Cerquilho': 
		cod_munic = 3511508
	if nome_munic == 'Cesário Lange': 
		cod_munic = 3511607
	if nome_munic == 'Charqueada': 
		cod_munic = 3511706
	if nome_munic == 'Chavantes': 
		cod_munic = 3557204
	if nome_munic == 'Clementina': 
		cod_munic = 3511904
	if nome_munic == 'Colina': 
		cod_munic = 3512001
	if nome_munic == 'Colômbia': 
		cod_munic = 3512100
	if nome_munic == 'Conchal': 
		cod_munic = 3512209
	if nome_munic == 'Conchas': 
		cod_munic = 3512308
	if nome_munic == 'Cordeirópolis': 
		cod_munic = 3512407
	if nome_munic == 'Coroados': 
		cod_munic = 3512506
	if nome_munic == 'Coronel Macedo': 
		cod_munic = 3512605
	if nome_munic == 'Corumbataí': 
		cod_munic = 3512704
	if nome_munic == 'Cosmópolis': 
		cod_munic = 3512803
	if nome_munic == 'Cosmorama': 
		cod_munic = 3512902
	if nome_munic == 'Cotia': 
		cod_munic = 3513009
	if nome_munic == 'Cravinhos': 
		cod_munic = 3513108
	if nome_munic == 'Cristais Paulista': 
		cod_munic = 3513207
	if nome_munic == 'Cruzália': 
		cod_munic = 3513306
	if nome_munic == 'Cruzeiro': 
		cod_munic = 3513405
	if nome_munic == 'Cubatão': 
		cod_munic = 3513504
	if nome_munic == 'Cunha': 
		cod_munic = 3513603
	if nome_munic == 'Descalvado': 
		cod_munic = 3513702
	if nome_munic == 'Diadema': 
		cod_munic = 3513801
	if nome_munic == 'Dirce Reis': 
		cod_munic = 3513850
	if nome_munic == 'Divinolândia': 
		cod_munic = 3513900
	if nome_munic == 'Dobrada': 
		cod_munic = 3514007
	if nome_munic == 'Dois Córregos': 
		cod_munic = 3514106
	if nome_munic == 'Dolcinópolis': 
		cod_munic = 3514205
	if nome_munic == 'Dourado': 
		cod_munic = 3514304
	if nome_munic == 'Dracena': 
		cod_munic = 3514403
	if nome_munic == 'Duartina': 
		cod_munic = 3514502
	if nome_munic == 'Dumont': 
		cod_munic = 3514601
	if nome_munic == 'Echaporã': 
		cod_munic = 3514700
	if nome_munic == 'Eldorado': 
		cod_munic = 3514809
	if nome_munic == 'Elias Fausto': 
		cod_munic = 3514908
	if nome_munic == 'Elisiário': 
		cod_munic = 3514924
	if nome_munic == 'Embaúba': 
		cod_munic = 3514957
	if nome_munic == 'Embu das Artes': 
		cod_munic = 3515004
	if nome_munic == 'Embu-Guaçu': 
		cod_munic = 3515103
	if nome_munic == 'Emilianópolis': 
		cod_munic = 3515129
	if nome_munic == 'Engenheiro Coelho': 
		cod_munic = 3515152
	if nome_munic == 'Espírito Santo do Pinhal': 
		cod_munic = 3515186
	if nome_munic == 'Espírito Santo do Turvo': 
		cod_munic = 3515194
	if nome_munic == 'Estiva Gerbi': 
		cod_munic = 3557303
	if nome_munic == 'Estrela do Norte': 
		cod_munic = 3515301
	if nome_munic == 'Estrela dOeste': 
		cod_munic = 3515202
	if nome_munic == 'Euclides da Cunha Paulista': 
		cod_munic = 3515350
	if nome_munic == 'Fartura': 
		cod_munic = 3515400
	if nome_munic == 'Fernando Prestes': 
		cod_munic = 3515608
	if nome_munic == 'Fernandópolis': 
		cod_munic = 3515509
	if nome_munic == 'Fernão': 
		cod_munic = 3515657
	if nome_munic == 'Ferraz de Vasconcelos': 
		cod_munic = 3515707
	if nome_munic == 'Flora Rica': 
		cod_munic = 3515806
	if nome_munic == 'Floreal': 
		cod_munic = 3515905
	if nome_munic == 'Flórida Paulista': 
		cod_munic = 3516002
	if nome_munic == 'Florínea': 
		cod_munic = 3516101
	if nome_munic == 'Franca': 
		cod_munic = 3516200
	if nome_munic == 'Francisco Morato': 
		cod_munic = 3516309
	if nome_munic == 'Franco da Rocha': 
		cod_munic = 3516408
	if nome_munic == 'Gabriel Monteiro': 
		cod_munic = 3516507
	if nome_munic == 'Gália': 
		cod_munic = 3516606
	if nome_munic == 'Garça': 
		cod_munic = 3516705
	if nome_munic == 'Gastão Vidigal': 
		cod_munic = 3516804
	if nome_munic == 'Gavião Peixoto': 
		cod_munic = 3516853
	if nome_munic == 'General Salgado': 
		cod_munic = 3516903
	if nome_munic == 'Getulina': 
		cod_munic = 3517000
	if nome_munic == 'Glicério': 
		cod_munic = 3517109
	if nome_munic == 'Guaiçara': 
		cod_munic = 3517208
	if nome_munic == 'Guaimbê': 
		cod_munic = 3517307
	if nome_munic == 'Guaíra': 
		cod_munic = 3517406
	if nome_munic == 'Guapiaçu': 
		cod_munic = 3517505
	if nome_munic == 'Guapiara': 
		cod_munic = 3517604
	if nome_munic == 'Guará': 
		cod_munic = 3517703
	if nome_munic == 'Guaraçaí': 
		cod_munic = 3517802
	if nome_munic == 'Guaraci': 
		cod_munic = 3517901
	if nome_munic == 'Guarani dOeste': 
		cod_munic = 3518008
	if nome_munic == 'Guarantã': 
		cod_munic = 3518107
	if nome_munic == 'Guararapes': 
		cod_munic = 3518206
	if nome_munic == 'Guararema': 
		cod_munic = 3518305
	if nome_munic == 'Guaratinguetá': 
		cod_munic = 3518404
	if nome_munic == 'Guareí': 
		cod_munic = 3518503
	if nome_munic == 'Guariba': 
		cod_munic = 3518602
	if nome_munic == 'Guarujá': 
		cod_munic = 3518701
	if nome_munic == 'Guarulhos': 
		cod_munic = 3518800
	if nome_munic == 'Guatapará': 
		cod_munic = 3518859
	if nome_munic == 'Guzolândia': 
		cod_munic = 3518909
	if nome_munic == 'Herculândia': 
		cod_munic = 3519006
	if nome_munic == 'Holambra': 
		cod_munic = 3519055
	if nome_munic == 'Hortolândia': 
		cod_munic = 3519071
	if nome_munic == 'Iacanga': 
		cod_munic = 3519105
	if nome_munic == 'Iacri': 
		cod_munic = 3519204
	if nome_munic == 'Iaras': 
		cod_munic = 3519253
	if nome_munic == 'Ibaté': 
		cod_munic = 3519303
	if nome_munic == 'Ibirá': 
		cod_munic = 3519402
	if nome_munic == 'Ibirarema': 
		cod_munic = 3519501
	if nome_munic == 'Ibitinga': 
		cod_munic = 3519600
	if nome_munic == 'Ibiúna': 
		cod_munic = 3519709
	if nome_munic == 'Icém': 
		cod_munic = 3519808
	if nome_munic == 'Iepê': 
		cod_munic = 3519907
	if nome_munic == 'Igaraçu do Tietê': 
		cod_munic = 3520004
	if nome_munic == 'Igarapava': 
		cod_munic = 3520103
	if nome_munic == 'Igaratá': 
		cod_munic = 3520202
	if nome_munic == 'Iguape': 
		cod_munic = 3520301
	if nome_munic == 'Ilha Comprida': 
		cod_munic = 3520426
	if nome_munic == 'Ilha Solteira': 
		cod_munic = 3520442
	if nome_munic == 'Ilhabela': 
		cod_munic = 3520400
	if nome_munic == 'Indaiatuba': 
		cod_munic = 3520509
	if nome_munic == 'Indiana': 
		cod_munic = 3520608
	if nome_munic == 'Indiaporã': 
		cod_munic = 3520707
	if nome_munic == 'Inúbia Paulista': 
		cod_munic = 3520806
	if nome_munic == 'Ipaussu': 
		cod_munic = 3520905
	if nome_munic == 'Iperó': 
		cod_munic = 3521002
	if nome_munic == 'Ipeúna': 
		cod_munic = 3521101
	if nome_munic == 'Ipiguá': 
		cod_munic = 3521150
	if nome_munic == 'Iporanga': 
		cod_munic = 3521200
	if nome_munic == 'Ipuã': 
		cod_munic = 3521309
	if nome_munic == 'Iracemápolis': 
		cod_munic = 3521408
	if nome_munic == 'Irapuã': 
		cod_munic = 3521507
	if nome_munic == 'Irapuru': 
		cod_munic = 3521606
	if nome_munic == 'Itaberá': 
		cod_munic = 3521705
	if nome_munic == 'Itaí': 
		cod_munic = 3521804
	if nome_munic == 'Itajobi': 
		cod_munic = 3521903
	if nome_munic == 'Itaju': 
		cod_munic = 3522000
	if nome_munic == 'Itanhaém': 
		cod_munic = 3522109
	if nome_munic == 'Itaoca': 
		cod_munic = 3522158
	if nome_munic == 'Itapecerica da Serra': 
		cod_munic = 3522208
	if nome_munic == 'Itapetininga': 
		cod_munic = 3522307
	if nome_munic == 'Itapeva': 
		cod_munic = 3522406
	if nome_munic == 'Itapevi': 
		cod_munic = 3522505
	if nome_munic == 'Itapira': 
		cod_munic = 3522604
	if nome_munic == 'Itapirapuã Paulista': 
		cod_munic = 3522653
	if nome_munic == 'Itápolis': 
		cod_munic = 3522703
	if nome_munic == 'Itaporanga': 
		cod_munic = 3522802
	if nome_munic == 'Itapuí': 
		cod_munic = 3522901
	if nome_munic == 'Itapura': 
		cod_munic = 3523008
	if nome_munic == 'Itaquaquecetuba': 
		cod_munic = 3523107
	if nome_munic == 'Itararé': 
		cod_munic = 3523206
	if nome_munic == 'Itariri': 
		cod_munic = 3523305
	if nome_munic == 'Itatiba': 
		cod_munic = 3523404
	if nome_munic == 'Itatinga': 
		cod_munic = 3523503
	if nome_munic == 'Itirapina': 
		cod_munic = 3523602
	if nome_munic == 'Itirapuã': 
		cod_munic = 3523701
	if nome_munic == 'Itobi': 
		cod_munic = 3523800
	if nome_munic == 'Itu': 
		cod_munic = 3523909
	if nome_munic == 'Itupeva': 
		cod_munic = 3524006
	if nome_munic == 'Ituverava': 
		cod_munic = 3524105
	if nome_munic == 'Jaborandi': 
		cod_munic = 3524204
	if nome_munic == 'Jaboticabal': 
		cod_munic = 3524303
	if nome_munic == 'Jacareí': 
		cod_munic = 3524402
	if nome_munic == 'Jaci': 
		cod_munic = 3524501
	if nome_munic == 'Jacupiranga': 
		cod_munic = 3524600
	if nome_munic == 'Jaguariúna': 
		cod_munic = 3524709
	if nome_munic == 'Jales': 
		cod_munic = 3524808
	if nome_munic == 'Jambeiro': 
		cod_munic = 3524907
	if nome_munic == 'Jandira': 
		cod_munic = 3525003
	if nome_munic == 'Jardinópolis': 
		cod_munic = 3525102
	if nome_munic == 'Jarinu': 
		cod_munic = 3525201
	if nome_munic == 'Jaú': 
		cod_munic = 3525300
	if nome_munic == 'Jeriquara': 
		cod_munic = 3525409
	if nome_munic == 'Joanópolis': 
		cod_munic = 3525508
	if nome_munic == 'João Ramalho': 
		cod_munic = 3525607
	if nome_munic == 'José Bon	ifácio': 
		cod_munic = 3525706
	if nome_munic == 'Júlio Mesquita': 
		cod_munic = 3525805
	if nome_munic == 'Jumirim': 
		cod_munic = 3525854
	if nome_munic == 'Jundiaí': 
		cod_munic = 3525904
	if nome_munic == 'Junqueirópolis': 
		cod_munic = 3526001
	if nome_munic == 'Juquiá': 
		cod_munic = 3526100
	if nome_munic == 'Juquitiba': 
		cod_munic = 3526209
	if nome_munic == 'Laranjal Paulista': 
		cod_munic = 3526407
	if nome_munic == 'Lavínia': 
		cod_munic = 3526506
	if nome_munic == 'Lavrinhas': 
		cod_munic = 3526605
	if nome_munic == 'Leme': 
		cod_munic = 3526704
	if nome_munic == 'Lençóis Paulista': 
		cod_munic = 3526803
	if nome_munic == 'Limeira': 
		cod_munic = 3526902
	if nome_munic == 'Lindóia': 
		cod_munic = 3527009
	if nome_munic == 'Lins': 
		cod_munic = 3527108
	if nome_munic == 'Lorena': 
		cod_munic = 3527207
	if nome_munic == 'Lourdes': 
		cod_munic = 3527256
	if nome_munic == 'Louveira': 
		cod_munic = 3527306
	if nome_munic == 'Lucélia': 
		cod_munic = 3527405
	if nome_munic == 'Luís Antônio': 
		cod_munic = 3527603
	if nome_munic == 'Luiziânia': 
		cod_munic = 3527702
	if nome_munic == 'Lupércio': 
		cod_munic = 3527801
	if nome_munic == 'Lutécia': 
		cod_munic = 3527900
	if nome_munic == 'Macatuba': 
		cod_munic = 3528007
	if nome_munic == 'Macaubal': 
		cod_munic = 3528106
	if nome_munic == 'Macedônia': 
		cod_munic = 3528205
	if nome_munic == 'Magda': 
		cod_munic = 3528304
	if nome_munic == 'Mairinque': 
		cod_munic = 3528403
	if nome_munic == 'Mairiporã': 
		cod_munic = 3528502
	if nome_munic == 'Manduri': 
		cod_munic = 3528601
	if nome_munic == 'Marabá Paulista': 
		cod_munic = 3528700
	if nome_munic == 'Maracaí': 
		cod_munic = 3528809
	if nome_munic == 'Marapoama': 
		cod_munic = 3528858
	if nome_munic == 'Mariápolis': 
		cod_munic = 3528908
	if nome_munic == 'Marília': 
		cod_munic = 3529005
	if nome_munic == 'Marinópolis': 
		cod_munic = 3529104
	if nome_munic == 'Martinópolis': 
		cod_munic = 3529203
	if nome_munic == 'Matão': 
		cod_munic = 3529302
	if nome_munic == 'Mauá': 
		cod_munic = 3529401
	if nome_munic == 'Mendonça': 
		cod_munic = 3529500
	if nome_munic == 'Meridiano': 
		cod_munic = 3529609
	if nome_munic == 'Mesópolis': 
		cod_munic = 3529658
	if nome_munic == 'Miguelópolis': 
		cod_munic = 3529708
	if nome_munic == 'Mineiros do Tietê': 
		cod_munic = 3529807
	if nome_munic == 'Mira Estrela': 
		cod_munic = 3530003
	if nome_munic == 'Miracatu': 
		cod_munic = 3529906
	if nome_munic == 'Mirandópolis': 
		cod_munic = 3530102
	if nome_munic == 'Mirante do Paranapanema': 
		cod_munic = 3530201
	if nome_munic == 'Mirassol': 
		cod_munic = 3530300
	if nome_munic == 'Mirassolândia': 
		cod_munic = 3530409
	if nome_munic == 'Mococa': 
		cod_munic = 3530508
	if nome_munic == 'Mogi das Cruzes': 
		cod_munic = 3530607
	if nome_munic == 'Mogi Guaçu': 
		cod_munic = 3530706
	if nome_munic == 'Mogi Mirim': 
		cod_munic = 3530805
	if nome_munic == 'Mombuca': 
		cod_munic = 3530904
	if nome_munic == 'Monções': 
		cod_munic = 3531001
	if nome_munic == 'Mongaguá': 
		cod_munic = 3531100
	if nome_munic == 'Monte Alegre do Sul': 
		cod_munic = 3531209
	if nome_munic == 'Monte Alto': 
		cod_munic = 3531308
	if nome_munic == 'Monte Aprazível': 
		cod_munic = 3531407
	if nome_munic == 'Monte Azul Paulista': 
		cod_munic = 3531506
	if nome_munic == 'Monte Castelo': 
		cod_munic = 3531605
	if nome_munic == 'Monte Mor': 
		cod_munic = 3531803
	if nome_munic == 'Morro Agudo': 
		cod_munic = 3531902
	if nome_munic == 'Morungaba': 
		cod_munic = 3532009
	if nome_munic == 'Motuca': 
		cod_munic = 3532058
	if nome_munic == 'Murutinga do Sul': 
		cod_munic = 3532108
	if nome_munic == 'Nantes': 
		cod_munic = 3532157
	if nome_munic == 'Narandiba': 
		cod_munic = 3532207
	if nome_munic == 'Natividade da Serra': 
		cod_munic = 3532306
	if nome_munic == 'Nazaré Paulista': 
		cod_munic = 3532405
	if nome_munic == 'Neves Paulista': 
		cod_munic = 3532504
	if nome_munic == 'Nhandeara': 
		cod_munic = 3532603
	if nome_munic == 'Nipoã': 
		cod_munic = 3532702
	if nome_munic == 'Nova Aliança': 
		cod_munic = 3532801
	if nome_munic == 'Nova Campina': 
		cod_munic = 3532827
	if nome_munic == 'Nova Canaã Paulista': 
		cod_munic = 3532843
	if nome_munic == 'Nova Europa': 
		cod_munic = 3532900
	if nome_munic == 'Nova Granada': 
		cod_munic = 3533007
	if nome_munic == 'Nova Guataporanga': 
		cod_munic = 3533106
	if nome_munic == 'Nova Independência': 
		cod_munic = 3533205
	if nome_munic == 'Nova Luzitânia': 
		cod_munic = 3533304
	if nome_munic == 'Nova Odessa': 
		cod_munic = 3533403
	if nome_munic == 'Novais': 
		cod_munic = 3533254
	if nome_munic == 'Novo Horizonte': 
		cod_munic = 3533502
	if nome_munic == 'Nuporanga': 
		cod_munic = 3533601
	if nome_munic == 'Ocauçu': 
		cod_munic = 3533700
	if nome_munic == 'Olímpia': 
		cod_munic = 3533908
	if nome_munic == 'Onda Verde': 
		cod_munic = 3534005
	if nome_munic == 'Oriente': 
		cod_munic = 3534104
	if nome_munic == 'Orindiúva': 
		cod_munic = 3534203
	if nome_munic == 'Orlândia': 
		cod_munic = 3534302
	if nome_munic == 'Osasco': 
		cod_munic = 3534401
	if nome_munic == 'Oscar Bressane': 
		cod_munic = 3534500
	if nome_munic == 'Osvaldo Cruz': 
		cod_munic = 3534609
	if nome_munic == 'Ourinhos': 
		cod_munic = 3534708
	if nome_munic == 'Ouro Verde': 
		cod_munic = 3534807
	if nome_munic == 'Ouroeste': 
		cod_munic = 3534757
	if nome_munic == 'Pacaembu': 
		cod_munic = 3534906
	if nome_munic == 'Palestina': 
		cod_munic = 3535002
	if nome_munic == 'Palmares Paulista': 
		cod_munic = 3535101
	if nome_munic == 'Palmeira dOeste': 
		cod_munic = 3535200
	if nome_munic == 'Palmital': 
		cod_munic = 3535309
	if nome_munic == 'Panorama': 
		cod_munic = 3535408
	if nome_munic == 'Paraguaçu Paulista': 
		cod_munic = 3535507
	if nome_munic == 'Paraibuna': 
		cod_munic = 3535606
	if nome_munic == 'Paraíso': 
		cod_munic = 3535705
	if nome_munic == 'Paranapanema': 
		cod_munic = 3535804
	if nome_munic == 'Paranapuã': 
		cod_munic = 3535903
	if nome_munic == 'Parapuã': 
		cod_munic = 3536000
	if nome_munic == 'Pardinho': 
		cod_munic = 3536109
	if nome_munic == 'Pariquera-Açu': 
		cod_munic = 3536208
	if nome_munic == 'Parisi': 
		cod_munic = 3536257
	if nome_munic == 'Patrocínio Paulista': 
		cod_munic = 3536307
	if nome_munic == 'Paulicéia': 
		cod_munic = 3536406
	if nome_munic == 'Paulínia': 
		cod_munic = 3536505
	if nome_munic == 'Paulistânia': 
		cod_munic = 3536570
	if nome_munic == 'Paulo de Faria': 
		cod_munic = 3536604
	if nome_munic == 'Pederneiras': 
		cod_munic = 3536703
	if nome_munic == 'Pedra Bela': 
		cod_munic = 3536802
	if nome_munic == 'Pedranópolis': 
		cod_munic = 3536901
	if nome_munic == 'Pedregulho': 
		cod_munic = 3537008
	if nome_munic == 'Pedreira': 
		cod_munic = 3537107
	if nome_munic == 'Pedrinhas Paulista': 
		cod_munic = 3537156
	if nome_munic == 'Pedro de Toledo': 
		cod_munic = 3537206
	if nome_munic == 'Penápolis': 
		cod_munic = 3537305
	if nome_munic == 'Pereira Barreto': 
		cod_munic = 3537404
	if nome_munic == 'Pereiras': 
		cod_munic = 3537503
	if nome_munic == 'Peruíbe': 
		cod_munic = 3537602
	if nome_munic == 'Piacatu': 
		cod_munic = 3537701
	if nome_munic == 'Piedade': 
		cod_munic = 3537800
	if nome_munic == 'Pilar do Sul': 
		cod_munic = 3537909
	if nome_munic == 'Pindamonhangaba': 
		cod_munic = 3538006
	if nome_munic == 'Pindorama': 
		cod_munic = 3538105
	if nome_munic == 'Pinhalzinho': 
		cod_munic = 3538204
	if nome_munic == 'Piquerobi': 
		cod_munic = 3538303
	if nome_munic == 'Piquete': 
		cod_munic = 3538501
	if nome_munic == 'Piracaia': 
		cod_munic = 3538600
	if nome_munic == 'Piracicaba': 
		cod_munic = 3538709
	if nome_munic == 'Piraju': 
		cod_munic = 3538808
	if nome_munic == 'Pirajuí': 
		cod_munic = 3538907
	if nome_munic == 'Pirangi': 
		cod_munic = 3539004
	if nome_munic == 'Pirapora do Bom Jesus': 
		cod_munic = 3539103
	if nome_munic == 'Pirapozinho': 
		cod_munic = 3539202
	if nome_munic == 'Pirassununga': 
		cod_munic = 3539301
	if nome_munic == 'Piratininga': 
		cod_munic = 3539400
	if nome_munic == 'Pitangueiras': 
		cod_munic = 3539509
	if nome_munic == 'Planalto': 
		cod_munic = 3539608
	if nome_munic == 'Platina': 
		cod_munic = 3539707
	if nome_munic == 'Poá': 
		cod_munic = 3539806
	if nome_munic == 'Poloni': 
		cod_munic = 3539905
	if nome_munic == 'Pompéia': 
		cod_munic = 3540002
	if nome_munic == 'Pongaí': 
		cod_munic = 3540101
	if nome_munic == 'Pontal': 
		cod_munic = 3540200
	if nome_munic == 'Pontalinda': 
		cod_munic = 3540259
	if nome_munic == 'Pontes Gestal': 
		cod_munic = 3540309
	if nome_munic == 'Populina': 
		cod_munic = 3540408
	if nome_munic == 'Porangaba': 
		cod_munic = 3540507
	if nome_munic == 'Porto Feliz': 
		cod_munic = 3540606
	if nome_munic == 'Porto Ferreira': 
		cod_munic = 3540705
	if nome_munic == 'Potim': 
		cod_munic = 3540754
	if nome_munic == 'Potirendaba': 
		cod_munic = 3540804
	if nome_munic == 'Pracinha': 
		cod_munic = 3540853
	if nome_munic == 'Pradópolis': 
		cod_munic = 3540903
	if nome_munic == 'Praia Grande': 
		cod_munic = 3541000
	if nome_munic == 'Pratânia': 
		cod_munic = 3541059
	if nome_munic == 'Presidente Alves': 
		cod_munic = 3541109
	if nome_munic == 'Presidente Bernardes': 
		cod_munic = 3541208
	if nome_munic == 'Presidente Epitácio': 
		cod_munic = 3541307
	if nome_munic == 'Presidente Prudente': 
		cod_munic = 3541406
	if nome_munic == 'Presidente Venceslau': 
		cod_munic = 3541505
	if nome_munic == 'Promissão': 
		cod_munic = 3541604
	if nome_munic == 'Quadra': 
		cod_munic = 3541653
	if nome_munic == 'Quatá': 
		cod_munic = 3541703
	if nome_munic == 'Queiroz': 
		cod_munic = 3541802
	if nome_munic == 'Queluz': 
		cod_munic = 3541901
	if nome_munic == 'Quintana': 
		cod_munic = 3542008
	if nome_munic == 'Rafard': 
		cod_munic = 3542107
	if nome_munic == 'Rancharia': 
		cod_munic = 3542206
	if nome_munic == 'Redenção da Serra': 
		cod_munic = 3542305
	if nome_munic == 'Regente Feijó': 
		cod_munic = 3542404
	if nome_munic == 'Reginópolis': 
		cod_munic = 3542503
	if nome_munic == 'Registro': 
		cod_munic = 3542602
	if nome_munic == 'Restinga': 
		cod_munic = 3542701
	if nome_munic == 'Ribeira': 
		cod_munic = 3542800
	if nome_munic == 'Ribeirão Bonito': 
		cod_munic = 3542909
	if nome_munic == 'Ribeirão Branco': 
		cod_munic = 3543006
	if nome_munic == 'Ribeirão Corrente': 
		cod_munic = 3543105
	if nome_munic == 'Ribeirão do Sul': 
		cod_munic = 3543204
	if nome_munic == 'Ribeirão dos Índios': 
		cod_munic = 3543238
	if nome_munic == 'Ribeirão Grande': 
		cod_munic = 3543253
	if nome_munic == 'Ribeirão Pires': 
		cod_munic = 3543303
	if nome_munic == 'Ribeirão Preto': 
		cod_munic = 3543402
	if nome_munic == 'R	ifaina': 
		cod_munic = 3543600
	if nome_munic == 'Rincão': 
		cod_munic = 3543709
	if nome_munic == 'Rinópolis': 
		cod_munic = 3543808
	if nome_munic == 'Rio Claro': 
		cod_munic = 3543907
	if nome_munic == 'Rio das Pedras': 
		cod_munic = 3544004
	if nome_munic == 'Rio Grande da Serra': 
		cod_munic = 3544103
	if nome_munic == 'Riolândia': 
		cod_munic = 3544202
	if nome_munic == 'Riversul': 
		cod_munic = 3543501
	if nome_munic == 'Rosana': 
		cod_munic = 3544251
	if nome_munic == 'Roseira': 
		cod_munic = 3544301
	if nome_munic == 'Rubiácea': 
		cod_munic = 3544400
	if nome_munic == 'Rubinéia': 
		cod_munic = 3544509
	if nome_munic == 'Sabino': 
		cod_munic = 3544608
	if nome_munic == 'Sagres': 
		cod_munic = 3544707
	if nome_munic == 'Sales': 
		cod_munic = 3544806
	if nome_munic == 'Sales Oliveira': 
		cod_munic = 3544905
	if nome_munic == 'Salesópolis': 
		cod_munic = 3545001
	if nome_munic == 'Salmourão': 
		cod_munic = 3545100
	if nome_munic == 'Saltinho': 
		cod_munic = 3545159
	if nome_munic == 'Salto': 
		cod_munic = 3545209
	if nome_munic == 'Salto de Pirapora': 
		cod_munic = 3545308
	if nome_munic == 'Salto Grande': 
		cod_munic = 3545407
	if nome_munic == 'Sandovalina': 
		cod_munic = 3545506
	if nome_munic == 'Santa Adélia': 
		cod_munic = 3545605
	if nome_munic == 'Santa Albertina': 
		cod_munic = 3545704
	if nome_munic == 'Santa Bárbara dOeste': 
		cod_munic = 3545803
	if nome_munic == 'Santa Branca': 
		cod_munic = 3546009
	if nome_munic == 'Santa Clara dOeste': 
		cod_munic = 3546108
	if nome_munic == 'Santa Cruz da Conceição': 
		cod_munic = 3546207
	if nome_munic == 'Santa Cruz da Esperança': 
		cod_munic = 3546256
	if nome_munic == 'Santa Cruz das Palmeiras': 
		cod_munic = 3546306
	if nome_munic == 'Santa Cruz do Rio Pardo': 
		cod_munic = 3546405
	if nome_munic == 'Santa Ernestina': 
		cod_munic = 3546504
	if nome_munic == 'Santa Fé do Sul': 
		cod_munic = 3546603
	if nome_munic == 'Santa Gertrudes': 
		cod_munic = 3546702
	if nome_munic == 'Santa Isabel': 
		cod_munic = 3546801
	if nome_munic == 'Santa Lúcia': 
		cod_munic = 3546900
	if nome_munic == 'Santa Maria da Serra': 
		cod_munic = 3547007
	if nome_munic == 'Santa Mercedes': 
		cod_munic = 3547106
	if nome_munic == 'Santa Rita do Passa Quatro': 
		cod_munic = 3547502
	if nome_munic == 'Santa Rita dOeste': 
		cod_munic = 3547403
	if nome_munic == 'Santa Rosa de Viterbo': 
		cod_munic = 3547601
	if nome_munic == 'Santa Salete': 
		cod_munic = 3547650
	if nome_munic == 'Santana da Ponte Pensa': 
		cod_munic = 3547205
	if nome_munic == 'Santana de Parnaíba': 
		cod_munic = 3547304
	if nome_munic == 'Santo Anastácio': 
		cod_munic = 3547700
	if nome_munic == 'Santo André': 
		cod_munic = 3547809
	if nome_munic == 'Santo Antônio da Alegria': 
		cod_munic = 3547908
	if nome_munic == 'Santo Antônio de Posse': 
		cod_munic = 3548005
	if nome_munic == 'Santo Antônio do Aracanguá': 
		cod_munic = 3548054
	if nome_munic == 'Santo Antônio do Jardim': 
		cod_munic = 3548104
	if nome_munic == 'Santo Antônio do Pinhal': 
		cod_munic = 3548203
	if nome_munic == 'Santo Expedito': 
		cod_munic = 3548302
	if nome_munic == 'Santópolis do Aguapeí': 
		cod_munic = 3548401
	if nome_munic == 'Santos': 
		cod_munic = 3548500
	if nome_munic == 'São Bento do Sapucaí': 
		cod_munic = 3548609
	if nome_munic == 'São Bernardo do Campo': 
		cod_munic = 3548708
	if nome_munic == 'São Caetano do Sul': 
		cod_munic = 3548807
	if nome_munic == 'São Carlos': 
		cod_munic = 3548906
	if nome_munic == 'São Francisco': 
		cod_munic = 3549003
	if nome_munic == 'São João da Boa Vista': 
		cod_munic = 3549102
	if nome_munic == 'São João das Duas Pontes': 
		cod_munic = 3549201
	if nome_munic == 'São João de Iracema': 
		cod_munic = 3549250
	if nome_munic == 'São João do Pau dAlho': 
		cod_munic = 3549300
	if nome_munic == 'São Joaquim da Barra': 
		cod_munic = 3549409
	if nome_munic == 'São José da Bela Vista': 
		cod_munic = 3549508
	if nome_munic == 'São José do Barreiro': 
		cod_munic = 3549607
	if nome_munic == 'São José do Rio Pardo': 
		cod_munic = 3549706
	if nome_munic == 'São José do Rio Preto': 
		cod_munic = 3549805
	if nome_munic == 'São José dos Campos': 
		cod_munic = 3549904
	if nome_munic == 'São Lourenço da Serra': 
		cod_munic = 3549953
	if nome_munic == 'São Luiz do Paraitinga': 
		cod_munic = 3550001
	if nome_munic == 'São Manuel': 
		cod_munic = 3550100
	if nome_munic == 'São Miguel Arcanjo': 
		cod_munic = 3550209
	if nome_munic == 'São Paulo': 
		cod_munic = 3550308
	if nome_munic == 'São Pedro': 
		cod_munic = 3550407
	if nome_munic == 'São Pedro do Turvo': 
		cod_munic = 3550506
	if nome_munic == 'São Roque': 
		cod_munic = 3550605
	if nome_munic == 'São Sebastião': 
		cod_munic = 3550704
	if nome_munic == 'São Sebastião da Grama': 
		cod_munic = 3550803
	if nome_munic == 'São Simão': 
		cod_munic = 3550902
	if nome_munic == 'São Vicente': 
		cod_munic = 3551009
	if nome_munic == 'Sarapuí': 
		cod_munic = 3551108
	if nome_munic == 'Sarutaiá': 
		cod_munic = 3551207
	if nome_munic == 'Sebastianópolis do Sul': 
		cod_munic = 3551306
	if nome_munic == 'Serra Azul': 
		cod_munic = 3551405
	if nome_munic == 'Serra Negra': 
		cod_munic = 3551603
	if nome_munic == 'Serrana': 
		cod_munic = 3551504
	if nome_munic == 'Sertãozinho': 
		cod_munic = 3551702
	if nome_munic == 'Sete Barras': 
		cod_munic = 3551801
	if nome_munic == 'Severínia': 
		cod_munic = 3551900
	if nome_munic == 'Silveiras': 
		cod_munic = 3552007
	if nome_munic == 'Socorro': 
		cod_munic = 3552106
	if nome_munic == 'Sorocaba': 
		cod_munic = 3552205
	if nome_munic == 'Sud Mennucci': 
		cod_munic = 3552304
	if nome_munic == 'Sumaré': 
		cod_munic = 3552403
	if nome_munic == 'Suzanápolis': 
		cod_munic = 3552551
	if nome_munic == 'Suzano': 
		cod_munic = 3552502
	if nome_munic == 'Tabapuã': 
		cod_munic = 3552601
	if nome_munic == 'Tabatinga': 
		cod_munic = 3552700
	if nome_munic == 'Taboão da Serra': 
		cod_munic = 3552809
	if nome_munic == 'Taciba': 
		cod_munic = 3552908
	if nome_munic == 'Taguaí': 
		cod_munic = 3553005
	if nome_munic == 'Taiaçu': 
		cod_munic = 3553104
	if nome_munic == 'Taiúva': 
		cod_munic = 3553203
	if nome_munic == 'Tambaú': 
		cod_munic = 3553302
	if nome_munic == 'Tanabi': 
		cod_munic = 3553401
	if nome_munic == 'Tapiraí': 
		cod_munic = 3553500
	if nome_munic == 'Tapiratiba': 
		cod_munic = 3553609
	if nome_munic == 'Taquaral': 
		cod_munic = 3553658
	if nome_munic == 'Taquaritinga': 
		cod_munic = 3553708
	if nome_munic == 'Taquarituba': 
		cod_munic = 3553807
	if nome_munic == 'Taquarivaí': 
		cod_munic = 3553856
	if nome_munic == 'Tarabai': 
		cod_munic = 3553906
	if nome_munic == 'Tarumã': 
		cod_munic = 3553955
	if nome_munic == 'Tatuí': 
		cod_munic = 3554003
	if nome_munic == 'Taubaté': 
		cod_munic = 3554102
	if nome_munic == 'Tejupá': 
		cod_munic = 3554201
	if nome_munic == 'Teodoro Sampaio': 
		cod_munic = 3554300
	if nome_munic == 'Terra Roxa': 
		cod_munic = 3554409
	if nome_munic == 'Tietê': 
		cod_munic = 3554508
	if nome_munic == 'Timburi': 
		cod_munic = 3554607
	if nome_munic == 'Torre de Pedra': 
		cod_munic = 3554656
	if nome_munic == 'Torrinha': 
		cod_munic = 3554706
	if nome_munic == 'Trabiju': 
		cod_munic = 3554755
	if nome_munic == 'Tremembé': 
		cod_munic = 3554805
	if nome_munic == 'Três Fronteiras': 
		cod_munic = 3554904
	if nome_munic == 'Tuiuti': 
		cod_munic = 3554953
	if nome_munic == 'Tupã': 
		cod_munic = 3555000
	if nome_munic == 'Tupi Paulista': 
		cod_munic = 3555109
	if nome_munic == 'Turiúba': 
		cod_munic = 3555208
	if nome_munic == 'Turmalina': 
		cod_munic = 3555307
	if nome_munic == 'Ubarana': 
		cod_munic = 3555356
	if nome_munic == 'Ubatuba': 
		cod_munic = 3555406
	if nome_munic == 'Ubirajara': 
		cod_munic = 3555505
	if nome_munic == 'Uchoa': 
		cod_munic = 3555604
	if nome_munic == 'União Paulista': 
		cod_munic = 3555703
	if nome_munic == 'Urânia': 
		cod_munic = 3555802
	if nome_munic == 'Uru': 
		cod_munic = 3555901
	if nome_munic == 'Urupês': 
		cod_munic = 3556008
	if nome_munic == 'Valentim Gentil': 
		cod_munic = 3556107
	if nome_munic == 'Valinhos': 
		cod_munic = 3556206
	if nome_munic == 'Valparaíso': 
		cod_munic = 3556305
	if nome_munic == 'Vargem': 
		cod_munic = 3556354
	if nome_munic == 'Vargem Grande do Sul': 
		cod_munic = 3556404
	if nome_munic == 'Vargem Grande Paulista': 
		cod_munic = 3556453
	if nome_munic == 'Várzea Paulista': 
		cod_munic = 3556503
	if nome_munic == 'Vera Cruz': 
		cod_munic = 3556602
	if nome_munic == 'Vinhedo': 
		cod_munic = 3556701
	if nome_munic == 'Viradouro': 
		cod_munic = 3556800
	if nome_munic == 'Vista Alegre do Alto': 
		cod_munic = 3556909
	if nome_munic == 'Vitória Brasil': 
		cod_munic = 3556958
	if nome_munic == 'Votorantim': 
		cod_munic = 3557006
	if nome_munic == 'Votuporanga': 
		cod_munic = 3557105
	if nome_munic == 'Zacarias': 
		cod_munic = 3557154



	user_data = {
		'Código do município' : cod_munic,
		#'Mês do início dos sintomas': mes_sintomas,
		'Idade': idade,
		'Sexo': cs_sexo,
		'Possui asma?': asma,
		'Possui cardiopatia?': cardiopatia,
		'Possui diabetes': diabetes,
		'Possui doenca hematologica': doenca_hematologica,
		'Possui doenca hepatica': doenca_hepatica,
		'Possui doenca neurologica': doenca_neurologica,
		'Possui doenca renal': doenca_renal,
		'Possui imunodepressao': imunodepressao,
		'Possui obesidade': obesidade,
		'Possui outros fatores de risco': outros_fatores_de_risco,
		'Possui pneumopatia': pneumopatia,
		'Está grávida?': puerpera,
		'Possui sindrome de down': sindrome_de_down
	}

	features = pd.DataFrame(user_data, index=[0])

	return features
	
user_input_variables = get_user_data()


btn_predict = st.sidebar.button("Realizar Predição")
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

if btn_predict:

	y_pred = random_forest.predict(X_test)

	# Avaliação do modelo

	predict = random_forest.predict(user_input_variables)

	if predict == 1:
		risco = 'alto'
	if predict == 0:
		risco = 'baixo'

	st.subheader('Previsão: ')
	st.write(user_input, ' possui risco ', risco, ' de morte por COVID-19')

	st.subheader('Acurácia do modelo')

	random_forest.score(X_train, y_train)
	acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
	result = print(round(acc_random_forest,2,), "%")

	st.write(acc_random_forest)


parametros = pd.DataFrame({'feature':X_train.columns,'Parametros':np.round(random_forest.feature_importances_,3)})
parametros = parametros.sort_values('Parametros',ascending=False).set_index('feature')

parametros.plot.bar()

# criando um dataframe



# # título
# st.title("Análise de vulnerabilidade COVID-19 no estado de São Paulo - By Daniel Herrera")

# # subtítulo
# st.markdown("Este é um Aplicativo utilizado para exibir a solução de Ciência de Dados para o problema de predição de Risco de morte caso a pessoa seja contaminada por Covid-19")

# st.sidebar.subheader("Insira os dados do para a análise")

# # mapeando dados do usuário para cada atributo
# cod_munic = st.sidebar.number_input("Código do município", value=data.codigo_ibge.mean())
# idade = st.sidebar.number_input("Idade", value=data.idade.mean())
# sexo = st.sidebar.number_input("Sexo", value=data.cs_sexo.mean())
# asma = st.sidebar.number_input("Possui asma?", value=data.asma.mean())
# cardiopata = st.sidebar.number_input("Possui cardiopatia?", value=data.cardiopatia.mean())
# diabetes = st.sidebar.number_input("Possui diabetes??", value=data.diabetes.mean())
# doenca_hematologica = st.sidebar.number_input("Possui doença hematológica?", value=data.doenca_hematologica.mean())
# doenca_hepatica = st.sidebar.number_input("Possui doença hepática?", value=data.doenca_hepatica.mean())
# doenca_neurologica = st.sidebar.number_input("Possui doença neurológica?", value=data.doenca_neurologica.mean())
# doenca_renal = st.sidebar.number_input("Possui doença renal?", value=data.doenca_renal.mean())
# imunodepressao = st.sidebar.number_input("Possui imunodepressao?", value=data.imunodepressao.mean())
# obesidade = st.sidebar.number_input("Possui obesidade?", value=data.obesidade.mean())
# outros_fatores_de_risco = st.sidebar.number_input("Possui outros fatores de risco?", value=data.outros_fatores_de_risco.mean())
# pneumopatia = st.sidebar.number_input("Possui pneumopatia?", value=data.pneumopatia.mean())
# puerpera = st.sidebar.number_input("Está grávida?", value=data.puerpera.mean())
# sindrome_de_down = st.sidebar.number_input("Possui sindrome de down?", value=data.sindrome_de_down.mean())


# # inserindo um botão na tela
# btn_predict = st.sidebar.button("Realizar Predição")

# # verifica se o botão foi acionado
# if btn_predict:
#     result = data.predict([[cod_munic,idade,sexo,asma,cardiopata,diabetes,doenca_hematologica,doenca_hepatica,doenca_neurologica,doenca_renal,imunodepressao,obesidade,outros_fatores_de_risco,pneumopatia,puerpera,sindrome_de_down]])
#     st.subheader("O Risco Previsto do Cliente é:")
#     result = result[0]
#     st.write(result)

# # verificando o dataset
# st.subheader("Selecionando as Variáveis de análise dos clientes")

# # atributos para serem exibidos por padrão
# defaultcols = ["anot_cadastrais","indice_inad","class_renda","saldo_contas"]

# # defindo atributos a partir do multiselect
# cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# # exibindo os top 8 registro do dataframe
# st.dataframe(data[cols].head(7))