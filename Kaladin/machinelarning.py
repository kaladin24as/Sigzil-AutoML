import pandas as pd
import numpy as np
from sklearn import *
from sklearn import linear_model
from sklearn.calibration import LabelEncoder, LinearSVC
from sklearn.cross_decomposition import PLSSVD, PLSCanonical, PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KDTree, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from linearfunctions import *
from mostrarmenus import * 
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, LassoCV, LassoLars, LogisticRegression, MultiTaskElasticNet, MultiTaskLasso, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, TweedieRegressor
from sklearn.preprocessing import LabelEncoder



# Aca nos movemos entre los menus de machine learning
def machine_learning(opcion_principal):
    
    # Va ajustado con el resultado obtenido en el menu principal
    if opcion_principal == "1":
    
        while True:
            X, y = preprocesar_datos()        
            mostrar_menu_machinelearning()
            subopcion_machinelearning = input("Selecciona una opcion: ")
            aprendizaje_supervisado(subopcion_machinelearning, X, y)
            aprendizaje_no_supervisado(subopcion_machinelearning)
            if subopcion_machinelearning == "3":
                print("AyudaXDDepurar")
            elif subopcion_machinelearning == "4":
                break
            elif subopcion_machinelearning == "5":
                exit()
            else:
                print("Opcion no valida.")

                
                
def preprocesar_datos(): 
    file_name = input("Introduce el nombre del archivo de entrenamiento del modelo: ")
    test_name = input("Introduce el nombre del archivo de prueba del modelo: ")          
    datos_por_limpiar = pd.read_csv(file_name)
    print("Datos originales:")
    print(datos_por_limpiar.head())
    columns_to_convert = input("¿Deseas convertir alguna columna categórica en valores numéricos? (yes/no): ")
    if columns_to_convert.lower() == "yes":
        column_name_to_convert = input("Introduce el nombre de la columna a convertir: ")
        datos_por_limpiar = convertir_columna_a_numerica(datos_por_limpiar, column_name_to_convert)
    
    # Establecer el umbral de correlacion significativa
    umbral_correlaciones_significativas = float(input("Introduce el umbral de correlación: "))
    datos_por_limpiar, columns_dropped, correlaciones_significativas = limpiar_datos(datos_por_limpiar, umbral_correlaciones_significativas, umbral_correlaciones_significativas)
    print("Datos limpios después de la conversión y la limpieza:")
    print(datos_por_limpiar.head())
        
    print("Columnas que han sido eliminadas:")
    print(columns_dropped) 
    print("\n Correlaciones significativas:")
    for correlacion in correlaciones_significativas:
        print(correlacion)
            
    print(datos_por_limpiar.info())
    # Solicitar al usuario el nombre de la variable objetivo
    target_variable = input("Introduce el nombre de la variable objetivo: ")  
        
    # Solicitar al usuario el nombre de las variables predictoras
    predictor_variables = input("Introduce el nombre de las variables predictoras, separadas por coma: ")
    predictor_variables = [var.strip() for var in predictor_variables.split(",")]
    # Crear la matriz de características X y el vector objetivo y
    X = datos_por_limpiar[predictor_variables]
    y = datos_por_limpiar[target_variable]
    return X, y
                 
            
######

def aprendizaje_supervisado(subopcion_machinelearning, X, y):
    if subopcion_machinelearning == "1":
        while True:
            mostrar_menu_aprendizaje_supervisado()
            opcion_aprendizaje_supervisado = input("Selecciona una opcion: ")
            as_lineal_models(X, y, opcion_aprendizaje_supervisado)
            as_lineal_and_quadratic_discriminant_analysis(X, y, opcion_aprendizaje_supervisado)
            as_kernel_ridge_regression(X, y, opcion_aprendizaje_supervisado)
            as_support_vector_machines(X, y, opcion_aprendizaje_supervisado)
            as_stochastic_gradient_descent(X, y, opcion_aprendizaje_supervisado)
            as_nearest_neighbors(X, y, opcion_aprendizaje_supervisado)
            as_gaussian_processes(X, y, opcion_aprendizaje_supervisado)
            as_cross_decomposition(X, y, opcion_aprendizaje_supervisado)
            as_naive_bayes(X, y, opcion_aprendizaje_supervisado)
            as_decision_trees(X, y, opcion_aprendizaje_supervisado)
            as_ensembles(X, y, opcion_aprendizaje_supervisado)
            as_multiclass_and_multioutput_algorithms(X, y, opcion_aprendizaje_supervisado)
            as_semi_supervised_learning(X, y, opcion_aprendizaje_supervisado)
            as_isotonic_regression(X, y, opcion_aprendizaje_supervisado)
            as_probabilistic_regression(X, y, opcion_aprendizaje_supervisado)
            as_neural_networks(X, y, opcion_aprendizaje_supervisado)
            if opcion_aprendizaje_supervisado == "18":
                break
            elif opcion_aprendizaje_supervisado == "19":
                exit()
            else:
                print("Opcion no valida.")


def aprendizaje_no_supervisado(subopcion_machinelearning):
    if subopcion_machinelearning == "2":
        while True:
            mostrar_menu_aprendizaje_nosupervisado()
            opcion_aprendizaje_nosupervisado = input("Selecciona una opcion: ")
#            ns_gaussian_mixture_models(opcion_aprendizaje_nosupervisado)
#            ns_manifold_learning(opcion_aprendizaje_nosupervisado)
#            ns_clustering(opcion_aprendizaje_nosupervisado)
#            ns_biclustering(opcion_aprendizaje_nosupervisado)
#            ns_decomposing_signals_in_components(opcion_aprendizaje_nosupervisado)
#            ns_covaraince_estimation(opcion_aprendizaje_nosupervisado)
#            ns_novelty_and_outlier_detection(opcion_aprendizaje_nosupervisado)
#            ns_density_estimation(opcion_aprendizaje_nosupervisado)
#            ns_neural_network_models(opcion_aprendizaje_nosupervisado)
            if opcion_aprendizaje_nosupervisado == "10": 
                break
            elif opcion_aprendizaje_nosupervisado == "11":
                exit()
            else:
                print("Opcion no valida.")
                
###################################################################
####################Opciones de aprendizaje supervisado############
###################################################################


###Usamos el metodo de regresion lineal. Investigar si hay otros
def as_lineal_models(X,y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "1":
        while True:
            mostrar_menu_linear_models()
            opcion_modelo_lineal = input("Selecciona una opcion: ")
            linear_regression(X,y,opcion_modelo_lineal)
            Ridge(X,y,opcion_modelo_lineal)
            RidgeClassifier(X,y,opcion_modelo_lineal)
            Lasso(X,y,opcion_modelo_lineal)
            LassoCVz(X,y,opcion_modelo_lineal)
            MultiTaskLassoz(X,y,opcion_modelo_lineal)
            ElasticNetCustom(X,y,opcion_modelo_lineal)
            MultiTaskElasticNetz(X,y,opcion_modelo_lineal)
            LassoLarsz(X,y,opcion_modelo_lineal)
            BayesianRidgez(X,y,opcion_modelo_lineal)
            ARDRegressionz(X,y,opcion_modelo_lineal)
            LogisticRegressionz(X,y,opcion_modelo_lineal)
            TweedieRegressorz(X,y,opcion_modelo_lineal)
            Perceptronz(X,y,opcion_modelo_lineal)
            PassiveAggressiveClassifierz(X,y,opcion_modelo_lineal)
            PassiveAggressiveRegressorz(X,y,opcion_modelo_lineal)
            if opcion_modelo_lineal == "17":
                break
            elif opcion_modelo_lineal == "18":
                exit()
            else:
                print("Opcion no valida.")           

###################Analisis Discriminante lineal y cuadratico##########
###Son dos modelos
def as_lineal_and_quadratic_discriminant_analysis(X,y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "2":
        while True:
            mostrar_menu_lineal_and_quadratic_discriminant_analysis()
            opcion_lqda = input("Selecciona una opcion: ")
            if opcion_lqda == "1":
                solver = input("Ingrese el solucionador a utilizar (‘svd’, ‘lsqr’, ‘eigen’): ").lower()
                if solver == 'svd':
                    shrinkage = None  # Si el solucionador es 'svd', no se usa la contracción
                else:
                    shrinkage = input("Ingrese el parámetro de contracción ('auto' o un valor flotante entre 0 y 1): ").lower()
                    if shrinkage != 'auto':
                        shrinkage = float(shrinkage)
                store_covariance = input("¿Calcular explícitamente la matriz de covarianza ponderada dentro de clase? (True/False): ").lower() == 'true'
                tol = float(input("Ingrese el umbral absoluto para considerar significativa una singularidad de X: "))
                covariance_estimator = None  # Cambiado a None para evitar el error
                
                # Crear el modelo de Análisis Discriminante Lineal con los hiperparámetros especificados por el usuario
                lda = LinearDiscriminantAnalysis(
                    solver=solver,
                    shrinkage=shrinkage,
                    store_covariance=store_covariance,
                    tol=tol,
                    covariance_estimator=covariance_estimator
                )
                
                # Ajustar el modelo a los datos de entrada
                lda.fit(X, y)
                print(lda.score(X, y))
                # También puedes imprimir otros atributos del modelo si lo deseas
                print("Coeficientes:", lda.coef_)
                print("Intercepción:", lda.intercept_)
                
                return lda
            
            elif opcion_lqda == "2":
                tol = float(input("Ingrese el umbral absoluto para considerar significativa una singularidad de X: "))

                # Crear el modelo de Análisis Discriminante Cuadrático con los hiperparámetros especificados por el usuario
                qda = QuadraticDiscriminantAnalysis(
                    tol=tol,
                )

                # Ajustar el modelo a los datos de entrada
                qda.fit(X, y)
                print(qda.score(X, y))
                # También puedes imprimir otros atributos del modelo si lo deseas

                return qda
            elif opcion_lqda == "3":
                break
            elif opcion_lqda == "4":
                exit()
            else:
                print("Opcion no valida.")
                
####################################    
def as_kernel_ridge_regression(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "3":
        print("No implementado")
        #alpha = float(input("Ingrese el valor del parámetro alpha (regularización): "))
        #kernel = input("Ingrese el tipo de kernel ('linear', 'rbf', 'poly', 'sigmoid'): ")
        #gamma = float(input("Ingrese el valor del parámetro gamma para el kernel (si corresponde): "))
        #degree = int(input("Ingrese el grado del polinomio para el kernel polinomial (si corresponde): "))
        #coef0 = float(input("Ingrese el valor del parámetro coef0 para el kernel polinomial o sigmoidal (si corresponde): "))

        #kr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        #kr.fit(X, y)
        #kr.score(X, y)
        
##########################################################        
    
def as_support_vector_machines(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "4":
        while True:
            mostrar_menu_support_vector_machines()
            opcion_svm = input("Selecciona una opcion: ")
            if opcion_svm == "1":
                LinearSVCz(X,y)
            NuSVC(X,y,opcion_svm)
            SVCz(X,y,opcion_svm)
            if opcion_svm == "4":
                break
            elif opcion_svm == "5":
                exit()
            else:
                print("Opcion no valida.")


def as_stochastic_gradient_descent(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "5":
        while True:
            mostrar_menu_stochastic_gradient_descent()
            opcion_sgd = input("Selecciona una opcion: ")
            SDGClasifier(X,y,opcion_sgd)
            SDGRegressor(X,y,opcion_sgd)
            SDGOneClassSVM(X,y,opcion_sgd)
            StandardScaler(X,y,opcion_sgd)
            if opcion_sgd == "5":
                break
            elif opcion_sgd == "6":
                exit()
            else:
                print("Opcion no valida.")        
        
        
    
def as_nearest_neighbors(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "6":
        while True:
            mostrar_menu_nearest_neighbors()
            opcion_knn = input("Selecciona una opcion: ")
            KNeighborsClassifierz(X,y,opcion_knn)
            KNeighborsRegressorz(X,y,opcion_knn)
            KDTreez(X,y,opcion_knn)
            RadiusNeighborsClassifierz(X,y,opcion_knn)
            RadiusNeighborsRegressorz(X,y,opcion_knn)
            if opcion_knn == "6":
                break
            elif opcion_knn == "7":
                exit()
            else:
                print("Opcion no valida.")
                    
        
###########        
        
def as_gaussian_processes(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "7":
        while True:
            mostrar_menu_gaussian_processes()
            opcion_gpr = input("Selecciona una opcion: ")
            GaussianProcessRegressorz(X,y,opcion_gpr)
            GaussianProcessClassifierz(X,y,opcion_gpr)
            if opcion_gpr == "3":
                break
            elif opcion_gpr == "4":
                exit()
            else:
                print("Opcion no valida.")
        
        
def as_cross_decomposition(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "8":
        while True:
            mostrar_menu_cross_decomposition()
            opcion_pls = input("Selecciona una opcion: ")
            PLSCanonicalz(X,y,opcion_pls)
            PLSSVDz(X,y,opcion_pls)
            PLSRegressionz(X,y,opcion_pls)
            if opcion_pls == "4":
                break
            elif opcion_pls == "5":
                exit()
            else:
                print("Opcion no valida.")

###########



def as_naive_bayes(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "9":
        while True:
            mostrar_menu_gaussian_naive_bayes
            opcion_nb = input("Selecciona una opcion: ")
            GaussianNBz(X,y,opcion_nb)
            MultinomialNBz(X,y,opcion_nb)
            ComplementNBz(X,y,opcion_nb)
            BernoulliNBz(X,y,opcion_nb)
            CategoricalNBz(X,y,opcion_nb)
            if opcion_nb == "6":
                break
            elif opcion_nb == "7":
                exit()
            else:
                print("Opcion no valida.")
                      
        
def as_decision_trees(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "10":
        while True:
            mostrar_menu_decision_trees()
            opcion_dt = input("Selecciona una opcion: ")
            DecisionTreeClassifierz(X,y,opcion_dt)
            DecisionTreeRegressorz(X,y,opcion_dt)
            if opcion_dt == "3":
                break
            elif opcion_dt == "4":
                exit()
            else:
                print("Opcion no valida.")
        

        
def as_ensembles(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "11":
        while True:
            mostrar_menu_ensembles()
            opcion_gb = input("Selecciona una opcion: ")
            GradientBoostingClassifierz(X,y,opcion_gb)
            GradientBoostingRegressorz(X,y,opcion_gb)
            HistGradientBoostingClassifierz(X,y,opcion_gb)
            HistGradientBoostingRegressorz(X,y,opcion_gb)
            RandomForestClassifierz(X,y,opcion_gb)
            RandomForestRegressorz(X,y,opcion_gb)
            ExtraTreesClassifierz(X,y,opcion_gb)
            ExtraTreesRegressorz(X,y,opcion_gb)
            VotingClassifierz(X,y,opcion_gb)
            if opcion_gb == "10":       
                break
            elif opcion_gb == "11":
                exit()
            else:
                print("Opcion no valida.") 
        
        
        
def as_multiclass_and_multioutput_algorithms(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "12":
        while True:
            mostrar_menu_multiclass()        
            opcion_rf = input("Selecciona una opcion: ")
            OneVsRestClassifierz(X,y,opcion_rf)
            OneVsOneClassifierz(X,y,opcion_rf)
            OutputCodeClassifierz(X,y,opcion_rf)
            MultiOutputRegressorz(X,y,opcion_rf)
            RegressorChainz(X,y,opcion_rf)
            if opcion_rf == "6":
                break
            elif opcion_rf == "7":
                exit()
            else:
                print("Opcion no valida.")
        
                
def as_semi_supervised_learning(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "14":
        while True:
            mostrar_menu_semisupervised_learning()
            opcion_ssf = input("Selecciona una opcion: ")
            SelfTrainingClassifierz(X,y,opcion_ssf)
            LabelPropagationz(X,y,opcion_ssf)
            LabelSpreadingz(X,y,opcion_ssf)
            if opcion_ssf == "5":
                break
            elif opcion_ssf == "6":
                exit()
            else:
                print("Opcion no valida.")
                            
def as_isotonic_regression(X, y, opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "15":
        isotonic_regression(X, y)

##########

##########

def as_probabilistic_regression(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "16":
        lr = linear_model.LogisticRegression()
        lr.fit(X, y)
        lr.score(X, y)
        print(lr.score(X, y))

def as_neural_networks(X, y,opcion_aprendizaje_supervisado):
    if opcion_aprendizaje_supervisado == "17":
        while True:
            mostrar_menu_neural_network_models()
            opcion_nnm = input("Selecciona una opcion: ")
            MLPClassifierz(X,y,opcion_nnm)
            MLPRegressorz(X,y,opcion_nnm)
            if opcion_nnm == "3":
                break
            elif opcion_nnm == "4":
                exit()
            else:
                print("Opcion no valida.")        

        
    
####Convertir columna a numerica

def convertir_columna_a_numerica(df, column_name_to_convert):
    """
    Convierte una columna categórica en valores numéricos utilizando LabelEncoder.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos.
    column_name (str): El nombre de la columna que se desea convertir.

    Retorna:
    DataFrame: El DataFrame con la columna convertida a valores numéricos si la columna existe.
    None: Si la columna especificada no existe en los datos.
    """
    if column_name_to_convert in df.columns:
        le = LabelEncoder()
        df[column_name_to_convert] = le.fit_transform(df[column_name_to_convert])
        return df
    else:
        print("La columna especificada no existe en los datos.")
        return None
   

#############
#Cargar el conjunto de datos de entrenamiento y de prueba.




def limpiar_datos(df, threshold, umbral_correlations_significatives):
    """
    Limpia los datos de un DataFrame eliminando columnas con valores nulos que superan un umbral
    y encuentra correlaciones significativas entre las columnas restantes.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos.
    threshold (float): El umbral de valores nulos para descartar columnas.
    umbral_significativo (float): El umbral de correlación significativa. Por defecto es 0.3.

    Retorna:
    DataFrame: Un DataFrame con las columnas que no superan el umbral de valores nulos.
    list: Una lista de columnas eliminadas.
    list: Una lista de tuplas que contienen las correlaciones significativas.
    """
    null_count = df.isnull().sum()
    null_percentage = (null_count / len(df)) 
    columns_to_drop = null_percentage[null_percentage > threshold].index
    columns_dropped = list(columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    
    # Rellenar espacios en blanco con el promedio de cada columna
    if len(columns_dropped) == 0:
        df = df.fillna(df.mean())
    
    columnas_numericas = df.select_dtypes(include=['number'])
    correlacion = columnas_numericas.corr()
    correlaciones_significativas = correlacion[(correlacion >= umbral_correlations_significatives) | (correlacion <= -umbral_correlations_significatives)]
    correlaciones_list = []
    for col1 in correlaciones_significativas.columns:
        for col2 in correlaciones_significativas.index:
            if col1 != col2:
                valor_correlacion = correlaciones_significativas.loc[col2, col1]
                if abs(valor_correlacion) >= umbral_correlations_significatives:
                    correlaciones_list.append((col1, col2, valor_correlacion))
    return df, columns_dropped, correlaciones_list


############Suport Vector Machines

def LinearSVCz(X, y):
    #penalty = input("Ingrese el tipo de penalización ('l1' o 'l2'): ")
    #loss = input("Ingrese el tipo de función de pérdida ('hinge' o 'squared_hinge'): ")
    #dual = bool(input("Seleccione el método de optimización (True o False): "))
    #tol = float(input("Ingrese el valor de tolerancia para el criterio de parada: "))
    #C = float(input("Ingrese el valor del parámetro de regularización C: "))
    #multi_class = input("Seleccione el esquema de clasificación multiclase ('ovr' o 'crammer_singer'): ")
    #fit_intercept_str = input("Indique si se debe ajustar el intercepto (True o False): ")
    #fit_intercept = fit_intercept_str.lower() == 'true'
    #intercept_scaling = float(input("Ingrese el valor de escala para el término de intercepción: "))
    #random_state = int(input("Ingrese la semilla para la generación de números aleatorios: "))
    #max_iter = int(input("Ingrese el número máximo de iteraciones: "))


    # Crear y ajustar el modelo LinearSVC
    #model = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, 
                      #multi_class=multi_class, fit_intercept=fit_intercept, 
                      #intercept_scaling=intercept_scaling, 
                      #random_state=random_state, max_iter=max_iter)
    model = LinearSVC()
    model.fit(X, y)
    model.score(X, y)
    print(model.score(X, y))

    
def SVCz(X, y, opcion_svm):
    if opcion_svm == "2":
        svc = SVC()
        svc.fit(X, y)
        svc.score(X, y)
        print(svc.score(X, y))
    
def NuSVC(X, y, opcion_svm):
    if opcion_svm == "3":
        svc = SVC(kernel='rbf')
        svc.fit(X, y)
        svc.score(X, y)
        print(svc.score(X, y))      
    
###################Stochastic Gradient Descent########################

def SDGClasifier(X, y, opcion_sgd):
    if opcion_sgd == "1":
        sgd = linear_model.SGDClassifier()
        sgd.fit(X, y)
        sgd.score(X, y)
        print(sgd.score(X, y))
    
def SDGRegressor(X, y, opcion_sgd):
    if opcion_sgd == "2":
        sgd = linear_model.SGDRegressor()
        sgd.fit(X, y)
        sgd.score(X, y)
        print(sgd.score(X, y))
    
    
def SDGOneClassSVM(X, y, opcion_sgd):
    if opcion_sgd == "3":
        sgd = linear_model.SGDOneClassSVM()
        sgd.fit(X, y)
        sgd.score(X, y)
        print(sgd.score(X, y))
        
def StandardScaler(X, y, opcion_sgd):
    if opcion_sgd == "4":
        sgd = linear_model.SGDRegressor()
        sgd.fit(X, y)
        sgd.score(X, y)
        print(sgd.score(X, y))

###################### Nearest Neighbors
## Documentar mas.
def KNeighborsClassifierz(X, y, opcion_knn):
    if opcion_knn == "1":
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        knn.score(X, y)
        print(knn.score(X, y))
    
def KNeighborsRegressorz(X, y, opcion_knn):
    if opcion_knn == "2":
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X, y)
        knn.score(X, y)
        print(knn.score(X, y))
    

def KDTreez(X, y, opcion_knn):
    if opcion_knn == "3":
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        kdt.fit(X, y)
        kdt.score(X, y)
        print(kdt.score(X, y))
    
    
#def BallTree  Terminar de programar ##

def RadiusNeighborsClassifierz(X, y, opcion_knn):
    if opcion_knn == "4":
        knn = RadiusNeighborsClassifier(radius=1.0)
        knn.fit(X, y)
        knn.score(X, y)
        print(knn.score(X, y))
    
def RadiusNeighborsRegressorz(X, y, opcion_knn):
    if opcion_knn == "5":
        knn = RadiusNeighborsRegressor(radius=1.0)
        knn.fit(X, y)
        knn.score(X, y)
        print(knn.score(X, y))
    

##### Gaussian Processes

def GaussianProcessRegressorz(X, y, opcion_gpr):
    if opcion_gpr == "1":
        gpr = GaussianProcessRegressor()
        gpr.fit(X, y)
        gpr.score(X, y)
        print(gpr.score(X, y))
    
def GaussianProcessClassifierz(X, y, opcion_gpr):
    if opcion_gpr == "2":
        gpr = GaussianProcessClassifier()
        gpr.fit(X, y)
        gpr.score(X, y)
        print(gpr.score(X, y))
    
############ Cross Decomposition

def PLSCanonicalz(X, y, opcion_pls):
    if opcion_pls == "1":
        components = input("Introduce el numero de componentes: ")
        pls = PLSCanonical(components=components)
        pls.fit(X, y)
        pls.score(X, y)
        print(pls.score(X, y))

def PLSSVDz(X, y, opcion_pls):
    if opcion_pls == "2":
        components = input("Introduce el numero de componentes: ")
        pls = PLSSVD(components=components)
        pls.fit(X, y)
        pls.score(X, y)
        print(pls.score(X, y))
    
def PLSRegressionz(X, y, opcion_pls):
    if opcion_pls == "3":
        components = input("Introduce el numero de componentes: ")
        pls = PLSRegression(components=components)
        pls.fit(X, y)
        pls.score(X, y)
        print(pls.score(X, y))
    
#########    Gausian Naive Bayes
def GaussianNBz(X, y, opcion_nb):
    if opcion_nb == "1":
        gnb = GaussianNB()
        gnb.fit(X, y)
        gnb.score(X, y)
        print(gnb.score(X, y))
    
def MultinomialNBz(X, y, opcion_nb):
    if opcion_nb == "2":
        gnb = MultinomialNB()
        gnb.fit(X, y)
        gnb.score(X, y)
        print(gnb.score(X, y))
    
def ComplementNBz(X, y, opcion_nb):
    if opcion_nb == "3":
        gnb = ComplementNB()
        gnb.fit(X, y)
        gnb.score(X, y)
        print(gnb.score(X, y))
    
def BernoulliNBz(X, y, opcion_nb):
    if opcion_nb == "4":
        gnb = BernoulliNB()
        gnb.fit(X, y)
        gnb.score(X, y)
        print(gnb.score(X, y))
    
def CategoricalNBz(X, y, opcion_nb):
    if opcion_nb == "5":
        gnb = CategoricalNB()
        gnb.fit(X, y)
        gnb.score(X, y)
        print(gnb.score(X, y))
    
########## Decision Trees

def DecisionTreeClassifierz(X, y, opcion_dt):
    if opcion_dt == "1":
        dt = DecisionTreeClassifier()
        dt.fit(X, y)
        dt.score(X, y)
        print(dt.score(X, y))
    
def DecisionTreeRegressorz(X, y, opcion_dt):
    if opcion_dt == "2":
        dt = DecisionTreeRegressor()
        dt.fit(X, y)
        dt.score(X, y)
        print(dt.score(X, y))
    
####### Ensembles: Gradient Boosting, Random Forests

def GradientBoostingClassifierz(X, y, opcion_gb):
    if opcion_gb == "1":
        gb = GradientBoostingClassifier()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
    
def GradientBoostingRegressorz(X, y, opcion_gb):
    if opcion_gb == "2":
        gb = GradientBoostingRegressor()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
    
    
def HistGradientBoostingClassifierz(X, y, opcion_gb):
    if opcion_gb == "3":
        gb = HistGradientBoostingClassifier()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
    
def HistGradientBoostingRegressorz(X, y, opcion_gb):
    if opcion_gb == "4":
        gb = HistGradientBoostingRegressor()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
    
def RandomForestClassifierz(X, y, opcion_gb):
    if opcion_gb == "5":
        gb = RandomForestClassifier()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
        
def RandomForestRegressorz(X, y, opcion_gb):
    if opcion_gb == "6":
        gb = RandomForestRegressor()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))

    
def ExtraTreesClassifierz(X, y, opcion_gb):
    if opcion_gb == "7":
        gb = ExtraTreeClassifier()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
    
def ExtraTreesRegressorz(X, y, opcion_gb):
    if opcion_gb == "8":
        gb = ExtraTreeRegressor()
        gb.fit(X, y)
        gb.score(X, y)
        print(gb.score(X, y))
            
def VotingClassifierz(X, y, opcion_gb):
    if opcion_gb == "9":
        print("Opcion no disponible")
#        classifiers = {
#            '1': LogisticRegression(),
#            '2': SVC(),
#            '3': DecisionTreeClassifier(),
#            '4': RandomForestClassifier()
#        }
#
#        estimators = []
#
#        print("Selecciona los clasificadores que deseas incluir:")
#        print("1. Regresión Logística")
#        print("2. Support Vector Machine")
#        print("3. Árbol de Decisión")
#        print("4. Random Forest")
#
#        choices = input("Ingresa los números separados por comas (por ejemplo, '1,2,3'): ").split(',')

#        for choice in choices:
#            if choice in classifiers:
#                estimators.append(('clf' + choice, classifiers[choice]))
#            else:
#                print(f"El clasificador {choice} no es una opción válida.")

#        if estimators:
#            gb = VotingClassifier(estimators)
#            gb.fit(X, y)
#            score = gb.score(X, y)
#            print("Accuracy:", score)
#        else:
#            print("No se han seleccionado clasificadores válidos.")

    
###########Multiclass. Complicado de manejar.

def OneVsRestClassifierz(X, y, opcion_rf):
    if opcion_rf == "1":
        estimator = select_estimator()
        rf = OneVsRestClassifier(estimator)
        rf.fit(X, y)
        score = rf.score(X, y)
        print(score)
    
def OneVsOneClassifierz(X, y, opcion_rf):
    if opcion_rf == "2":
        rf = OneVsOneClassifier()
        rf.fit(X, y)
        rf.score(X, y)
        print(rf.score(X, y))
    
def OutputCodeClassifierz(X, y, opcion_rf):
    if opcion_rf == "3":
        rf = OutputCodeClassifier()
        rf.fit(X, y)
        rf.score(X, y)
        print(rf.score(X, y))
    
def MultiOutputRegressorz(X, y, opcion_rf):
    if opcion_rf == "4":
        rf = MultiOutputRegressor()
        rf.fit(X, y)
        rf.score(X, y)
        print(rf.score(X, y))
    
def RegressorChainz(X, y, opcion_rf):
    if opcion_rf == "5":
        rf = RegressorChain()
        rf.fit(X, y)
        rf.score(X, y)
        print(rf.score(X, y))

##########Semi-Supervised Learning

def SelfTrainingClassifierz(X, y, opcion_ssf):
    if opcion_ssf == "1":
        base_estimator = input("Ingrese el estimador base (debe implementar fit y predict_proba): ")
        threshold = float(input("Ingrese el umbral de decisión (threshold) (debe estar en [0, 1))): "))
        criterion = input("Ingrese el criterio de selección ('threshold' o 'k_best'): ")
        k_best = int(input("Ingrese la cantidad de muestras a agregar en cada iteración (k_best): "))
        max_iter = int(input("Ingrese el número máximo de iteraciones permitidas (max_iter) (deje en blanco para 10): ") or 10)
        verbose = input("¿Desea habilitar la salida detallada? ('True' o 'False', deje en blanco para 'False'): ").lower() == "true"
        
        # Crear y ajustar el modelo SelfTrainingClassifier
        model = SelfTrainingClassifier(base_estimator=base_estimator, threshold=threshold, criterion=criterion, 
                                    k_best=k_best, max_iter=max_iter, verbose=verbose)
        model.fit(X, y)
        score = model.score(X, y)
        print(score)
        return model

    
def LabelPropagationz(X, y, opcion_ssf):
    if opcion_ssf == "3":
        kernel = input("Ingrese el tipo de kernel ('knn' o 'rbf'): ")
        gamma = float(input("Ingrese el valor de gamma para el kernel RBF (gamma): "))
        n_neighbors = int(input("Ingrese el número de vecinos para el kernel KNN (n_neighbors): "))
        max_iter = int(input("Ingrese el número máximo de iteraciones permitidas (max_iter) (deje en blanco para 1000): ") or 1000)
        tol = float(input("Ingrese la tolerancia para la convergencia (tol) (deje en blanco para 0.001): ") or 0.001)
        n_jobs = input("Ingrese el número de trabajos en paralelo (n_jobs) (deje en blanco para None): ")
        if n_jobs != '':
            n_jobs = int(n_jobs)
        else:
            n_jobs = None
        
        # Crear y ajustar el modelo LabelPropagation
        model = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        model.fit(X, y)
        score = model.score(X, y)
        print(score)
        
        return model
def LabelSpreadingz(X, y, opcion_ssf):  
    if opcion_ssf == "4":
        kernel = input("Ingrese el tipo de kernel ('knn' o 'rbf'): ")
        gamma = float(input("Ingrese el valor de gamma para el kernel RBF (gamma): "))
        n_neighbors = int(input("Ingrese el número de vecinos para el kernel KNN (n_neighbors): "))
        alpha = float(input("Ingrese el factor de enganche suave (alpha) (deje en blanco para 0.2): ") or 0.2)
        max_iter = int(input("Ingrese el número máximo de iteraciones permitidas (max_iter) (deje en blanco para 30): ") or 30)
        tol = float(input("Ingrese la tolerancia para la convergencia (tol) (deje en blanco para 0.001): ") or 0.001)
        n_jobs = input("Ingrese el número de trabajos en paralelo (n_jobs) (deje en blanco para None): ")
        if n_jobs != '':
            n_jobs = int(n_jobs)
        else:
            n_jobs = None
        
        # Crear y ajustar el modelo LabelSpreading
        model = LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        model.fit(X, y)
        score = model.score(X, y)
        print(score)
        
        return model
    
########### Isotonic Regression

def isotonic_regression(X, y):
    y_min = float(input("Ingrese el límite inferior en el valor predicho (y_min) (deje en blanco para -inf): ") or "-inf")
    y_max = float(input("Ingrese el límite superior en el valor predicho (y_max) (deje en blanco para +inf): ") or "inf")
    increasing = input("¿Desea que las predicciones aumenten con X? ('True' o 'False', deje en blanco para 'True'): ").lower()
    increasing = True if increasing == "true" or not increasing else False
    out_of_bounds = input("¿Cómo desea manejar los valores de X fuera del dominio de entrenamiento durante la predicción? ('nan', 'clip', 'raise', deje en blanco para 'nan'): ").lower()
    out_of_bounds = out_of_bounds if out_of_bounds in ['nan', 'clip', 'raise'] else 'nan'

    # Crear y ajustar el modelo IsotonicRegression
    model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=increasing, out_of_bounds=out_of_bounds)
    model.fit(X, y)
    score = model.score(X, y)

    print("Puntaje de precisión del modelo:", score)
    
############## Neural Network Models (Supervised)

def MLPClassifierz(X, y, opcion_nnm):
    if opcion_nnm == "1":
        hidden_layer_sizes = tuple(int(x) for x in input("Ingrese el tamaño de las capas ocultas (separadas por espacios): ").split())
        activation = input("Ingrese la función de activación ('identity', 'logistic', 'tanh', 'relu'): ")
        solver = input("Ingrese el solucionador ('lbfgs', 'sgd', 'adam'): ")
        alpha = float(input("Ingrese el valor de alpha para la regularización L2: "))
        batch_size = input("Ingrese el tamaño del lote ('auto' o un entero): ")
        learning_rate = input("Ingrese la tasa de aprendizaje ('constant', 'invscaling', 'adaptive'): ")
        learning_rate_init = float(input("Ingrese la tasa de aprendizaje inicial: "))
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        random_state = int(input("Ingrese la semilla para la generación de números aleatorios: "))

        # Crear y ajustar el modelo MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init, max_iter=max_iter,
                            random_state=random_state)
        model.fit(X, y)
        score = model.score(X, y)

        print("Precisión del modelo:", score)

def MLPRegressorz(X, y, opcion_nnm):
    if opcion_nnm == "2":
        hidden_layer_sizes = tuple(int(x) for x in input("Ingrese el tamaño de las capas ocultas (separadas por espacios): ").split())
        activation = input("Ingrese la función de activación ('identity', 'logistic', 'tanh', 'relu'): ")
        solver = input("Ingrese el solucionador ('lbfgs', 'sgd', 'adam'): ")
        alpha = float(input("Ingrese el valor de alpha para la regularización L2: "))
        batch_size = input("Ingrese el tamaño del lote ('auto' o un entero): ")
        learning_rate = input("Ingrese la tasa de aprendizaje ('constant', 'invscaling', 'adaptive'): ")
        learning_rate_init = float(input("Ingrese la tasa de aprendizaje inicial: "))
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        random_state = input("Ingrese la semilla para la generación de números aleatorios (deje en blanco para None): ")
        random_state = int(random_state) if random_state.strip() else None

        # Crear y ajustar el modelo MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init, max_iter=max_iter,
                            random_state=random_state)
        model.fit(X, y)
        score = model.score(X, y)


        print("Puntaje de precisión del modelo:", score)
    
###### Funciones lineales
def linear_regression(X, y, opcion_modelo_lineal):
    if opcion_modelo_lineal == "1":
        print("1. Modelo lineal básico")
        print("2. Modelo lineal con coeficientes positivos")
        choice = input("Elige una opcion: ")
        if choice == "1":
            lr = linear_model.LinearRegression(positive=False)
            lr.fit(X, y)
            score = lr.score(X, y)
            print("Coeficiente de determinación (R^2):", score)
            
        elif choice == "2":
            lr = linear_model.LinearRegression(positive=True)
            lr.fit(X, y)
            score = lr.score(X, y)
            print("Coeficiente de determinación (R^2):", score)

##Depurar funcion. Falta analisis.
def Ridge(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "2":
        alpha = float(input("Ingrese el valor de alpha: "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        tol = 0.0001
        rg = linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept, tol=tol)
        rg.fit(X, y)
        score = rg.score(X, y)
        print("Coeficientes:", rg.coef_)
        print("Intercepción:", rg.intercept_)
        print("Score:", score)
        return rg
    

def RidgeClassifier(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "3":
        alpha = float(input("Ingrese el valor de alpha: "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        tol = 0.0001
        rgc = linear_model.RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, tol=tol)
        rgc.fit(X, y)
        score = rgc.score(X, y)
        print("Coeficientes:", rgc.coef_)
        print("Intercepción:", rgc.intercept_)
        print("Score:", score)
        return rgc
    
def Lasso(X, y, opcion_modelo_lineal):
    if opcion_modelo_lineal == "4":
        alpha = float(input("Ingrese el valor de alpha (debe ser un número no negativo): "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        precompute = input("¿Desea precomputar la matriz Gram para acelerar los cálculos? (True/False): ").lower() == 'true'
        copy_X = input("¿Desea copiar los datos de entrada? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        tol = float(input("Ingrese la tolerancia para la optimización: "))
        positive = input("¿Desea forzar que los coeficientes sean positivos? (True/False): ").lower() == 'true'
        random_state = None  # Opcional: puede permitir al usuario especificar una semilla aleatoria si lo desea
        selection = input("Seleccione el método de selección ('cyclic' o 'random'): ")

        # Crear el modelo Lasso con los parámetros especificados
        lasso = linear_model.Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            positive=positive,
            random_state=random_state,
            selection=selection
        )
        # Ajustar el modelo a los datos de entrada
        lasso.fit(X, y)

        # Calcular el coeficiente de determinación (score)
        score = lasso.score(X, y)
        # Imprimir los resultados
        print("Coeficientes:", lasso.coef_)
        print("Intercepción:", lasso.intercept_)
        print("Score:", score)
        return lasso    
    

def LassoCVz(X, y, opcion_modelo_lineal):
    if opcion_modelo_lineal == "5":
        # Solicitar al usuario los hiperparámetros específicos para LassoCV
        eps = float(input("Ingrese el valor de eps (longitud del camino): "))
        n_alphas = int(input("Ingrese el número de alphas a lo largo del camino: "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        precompute = input("¿Desea precomputar la matriz Gram para acelerar los cálculos? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        tol = float(input("Ingrese la tolerancia para la optimización: "))
        cv = int(input("Ingrese el número de divisiones de validación cruzada: "))
        verbose = input("¿Desea mostrar información detallada durante el ajuste? (True/False): ").lower() == 'true'
        n_jobs = int(input("Ingrese el número de trabajos para paralelizar el ajuste (-1 para utilizar todos los núcleos disponibles): "))
        positive = input("¿Desea forzar que los coeficientes sean positivos? (True/False): ").lower() == 'true'
        random_state = int(input("Ingrese la semilla aleatoria para la generación de números aleatorios (o deje en blanco para no especificar una semilla): ") or None)
        selection = input("Seleccione el método de selección ('cyclic' o 'random'): ")

        # Crear el modelo LassoCV con los hiperparámetros especificados por el usuario
        lasso_cv = LassoCV(
            eps=eps,
            n_alphas=n_alphas,
            fit_intercept=fit_intercept,
            precompute=precompute,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            selection=selection
        )
        
        # Ajustar el modelo a los datos de entrada
        lasso_cv.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = lasso_cv.score(X, y)
        print("Puntaje del modelo LassoCV:", score)
        print("Alpha óptimo:", lasso_cv.alpha_)
        print("Coeficientes:", lasso_cv.coef_)
        
        return lasso_cv
    
def MultiTaskLassoz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "6":
        # Solicitar al usuario los hiperparámetros específicos para MultiTaskLasso
        alpha = float(input("Ingrese el valor de alpha (debe ser un número no negativo): "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        copy_X = input("¿Desea copiar los datos de entrada? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        tol = float(input("Ingrese la tolerancia para la optimización: "))
        warm_start = input("¿Desea reutilizar la solución de la llamada anterior para inicializar el ajuste? (True/False): ").lower() == 'true'
        random_state = int(input("Ingrese la semilla aleatoria para la generación de números aleatorios (o deje en blanco para no especificar una semilla): ") or None)
        selection = input("Seleccione el método de selección ('cyclic' o 'random'): ")

        # Crear el modelo MultiTaskLasso con los hiperparámetros especificados por el usuario
        multi_task_lasso = MultiTaskLasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            random_state=random_state,
            selection=selection
        )
        
        # Ajustar el modelo a los datos de entrada
        multi_task_lasso.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = multi_task_lasso.score(X, y)
        print("Puntaje del modelo MultiTaskLasso:", score)        
        print("Alpha utilizado:", multi_task_lasso.alpha_)
        print("Coeficientes:", multi_task_lasso.coef_)
        print("Intercepción:", multi_task_lasso.intercept_)
        
        return multi_task_lasso
    
def ElasticNetCustom(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "7":

        # Solicitar al usuario los hiperparámetros específicos para ElasticNet
        alpha = float(input("Ingrese el valor de alpha (debe ser un número no negativo): "))
        l1_ratio = float(input("Ingrese el valor de l1_ratio (entre 0 y 1): "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        precompute = input("¿Desea utilizar una matriz de Gram precalculada para acelerar los cálculos? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        copy_X = input("¿Desea copiar los datos de entrada? (True/False): ").lower() == 'true'
        tol = float(input("Ingrese la tolerancia para la optimización: "))
        warm_start = input("¿Desea reutilizar la solución de la llamada anterior para inicializar el ajuste? (True/False): ").lower() == 'true'
        positive = input("¿Desea forzar que los coeficientes sean positivos? (True/False): ").lower() == 'true'
        random_state = int(input("Ingrese la semilla aleatoria para la generación de números aleatorios (o deje en blanco para no especificar una semilla): ") or None)
        selection = input("Seleccione el método de selección ('cyclic' o 'random'): ")

        # Crear el modelo ElasticNet con los hiperparámetros especificados por el usuario
        elastic_net = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            precompute=precompute,
            max_iter=max_iter,
            copy_X=copy_X,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection
        )
        
        # Ajustar el modelo a los datos de entrada
        elastic_net.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = elastic_net.score(X, y)
        print("Puntaje del modelo ElasticNet:", score)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Coeficientes:", elastic_net.coef_)
        print("Intercepción:", elastic_net.intercept_)
        
        return elastic_net
    
def MultiTaskElasticNetz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "8":
        alpha = float(input("Ingrese el valor de alpha (debe ser un número no negativo): "))
        l1_ratio = float(input("Ingrese el valor de l1_ratio (entre 0 y 1): "))
        fit_intercept = input("¿Desea ajustar la intercepción? (True/False): ").lower() == 'true'
        copy_X = input("¿Desea copiar los datos de entrada? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        tol = float(input("Ingrese la tolerancia para la optimización: "))
        warm_start = input("¿Desea reutilizar la solución de la llamada anterior para inicializar el ajuste? (True/False): ").lower() == 'true'
        random_state = int(input("Ingrese la semilla aleatoria para la generación de números aleatorios (o deje en blanco para no especificar una semilla): ") or None)
        selection = input("Seleccione el método de selección ('cyclic' o 'random'): ")

        # Crear el modelo MultiTaskElasticNet con los hiperparámetros especificados por el usuario
        multi_elastic_net = MultiTaskElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            random_state=random_state,
            selection=selection
        )
        
        # Ajustar el modelo a los datos de entrada
        multi_elastic_net.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = multi_elastic_net.score(X, y)
        print("Puntaje del modelo MultiTaskElasticNet:", score)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Alpha utilizado:", multi_elastic_net.alpha_)
        print("L1 ratio utilizado:", multi_elastic_net.l1_ratio_)
        print("Coeficientes:", multi_elastic_net.coef_)
        print("Intercepción:", multi_elastic_net.intercept_)
        
        return multi_elastic_net
    
def LassoLarsz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "9":
        alpha = float(input("Ingrese el valor de alpha (debe ser un número no negativo): "))
        fit_intercept = input("¿Desea calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        verbose = input("¿Desea activar la salida detallada? (True/False): ").lower() == 'true'
        precompute = input("¿Desea precalcular la matriz Gram para acelerar los cálculos? (True/False/auto): ").lower()
        max_iter = int(input("Ingrese el número máximo de iteraciones a realizar: "))
        eps = float(input("Ingrese el valor de epsilon para la regularización de precisión de la máquina: "))
        copy_X = input("¿Desea copiar la matriz de entrada? (True/False): ").lower() == 'true'
        fit_path = input("¿Desea almacenar el camino completo en el atributo coef_path_? (True/False): ").lower() == 'true'
        positive = input("¿Desea restringir los coeficientes para que sean >= 0? (True/False): ").lower() == 'true'
        jitter = input("¿Desea agregar un parámetro de ruido uniforme a los valores de y? (None o un número): ")
        random_state = int(input("Ingrese la semilla aleatoria para la generación de números aleatorios (o deje en blanco para no especificar una semilla): ") or None)
        
        # Crear el modelo LassoLars con los hiperparámetros especificados por el usuario
        lasso_lars = LassoLars(
            alpha=alpha,
            fit_intercept=fit_intercept,
            verbose=verbose,
            precompute=precompute,
            max_iter=max_iter,
            eps=eps,
            copy_X=copy_X,
            fit_path=fit_path,
            positive=positive,
            jitter=jitter,
            random_state=random_state
        )
        
        # Ajustar el modelo a los datos de entrada
        lasso_lars.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = lasso_lars.score(X, y)
        print("Puntaje del modelo LassoLars:", score)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Coeficientes:", lasso_lars.coef_)
        print("Intercepción:", lasso_lars.intercept_)
        
        return lasso_lars
    
def BayesianRidgez(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "10":
        max_iter = int(input("Ingrese el número máximo de iteraciones sobre el conjunto de datos (o deje en blanco para el valor predeterminado): ") or None)
        tol = float(input("Ingrese el valor de tolerancia para la convergencia del modelo: "))
        alpha_1 = float(input("Ingrese el valor de alpha_1 para la distribución Gamma prior sobre el parámetro alpha: "))
        alpha_2 = float(input("Ingrese el valor de alpha_2 para la distribución Gamma prior sobre el parámetro alpha: "))
        lambda_1 = float(input("Ingrese el valor de lambda_1 para la distribución Gamma prior sobre el parámetro lambda: "))
        lambda_2 = float(input("Ingrese el valor de lambda_2 para la distribución Gamma prior sobre el parámetro lambda: "))
        alpha_init = float(input("Ingrese el valor inicial para alpha (o deje en blanco para el valor predeterminado): ") or None)
        lambda_init = float(input("Ingrese el valor inicial para lambda (o deje en blanco para el valor predeterminado): ") or None)
        compute_score = input("¿Desea calcular el puntaje del modelo en cada iteración de la optimización? (True/False): ").lower() == 'true'
        fit_intercept = input("¿Desea calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        copy_X = input("¿Desea copiar la matriz de entrada? (True/False): ").lower() == 'true'
        verbose = input("¿Desea activar el modo detallado al ajustar el modelo? (True/False): ").lower() == 'true'
        
        # Crear el modelo BayesianRidge con los hiperparámetros especificados por el usuario
        bayesian_ridge = BayesianRidge(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose
        )
        
        # Ajustar el modelo a los datos de entrada
        bayesian_ridge.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = bayesian_ridge.score(X, y)
        print("Puntaje del modelo BayesianRidge:", score)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Coeficientes:", bayesian_ridge.coef_)
        print("Intercepción:", bayesian_ridge.intercept_)
        
        return bayesian_ridge
    
def ARDRegressionz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "11":
        max_iter = int(input("Ingrese el número máximo de iteraciones: ") or None)
        tol = float(input("Ingrese el valor de tolerancia para la convergencia del modelo: "))
        alpha_1 = float(input("Ingrese el valor de alpha_1 para la distribución Gamma prior sobre el parámetro alpha: "))
        alpha_2 = float(input("Ingrese el valor de alpha_2 para la distribución Gamma prior sobre el parámetro alpha: "))
        lambda_1 = float(input("Ingrese el valor de lambda_1 para la distribución Gamma prior sobre el parámetro lambda: "))
        lambda_2 = float(input("Ingrese el valor de lambda_2 para la distribución Gamma prior sobre el parámetro lambda: "))
        compute_score = input("¿Desea calcular la función objetivo en cada paso del modelo? (True/False): ").lower() == 'true'
        threshold_lambda = float(input("Ingrese el umbral para eliminar pesos con alta precisión del cálculo: "))
        fit_intercept = input("¿Desea calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        copy_X = input("¿Desea copiar la matriz de entrada? (True/False): ").lower() == 'true'
        verbose = input("¿Desea activar el modo detallado al ajustar el modelo? (True/False): ").lower() == 'true'
        
        # Crear el modelo ARDRegression con los hiperparámetros especificados por el usuario
        ard_regression = ARDRegression(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=compute_score,
            threshold_lambda=threshold_lambda,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose
        )
        
        # Ajustar el modelo a los datos de entrada
        ard_regression.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = ard_regression.score(X, y)
        print("Puntaje del modelo ARDRegression:", score)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Coeficientes:", ard_regression.coef_)
        print("Intercepción:", ard_regression.intercept_)
        
        return ard_regression
    
def LogisticRegressionz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "12":
        penalty = input("Ingrese el tipo de penalización ('l1', 'l2', 'elasticnet' o None): ")
        dual = input("¿Utilizar formulación dual? (True/False): ").lower() == 'true'
        tol = float(input("Ingrese el valor de tolerancia para la convergencia del modelo: "))
        C = float(input("Ingrese el valor de la inversa de la fuerza de regularización C: "))
        fit_intercept = input("¿Calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        intercept_scaling = float(input("Ingrese el valor de escala para la intercepción: "))
        class_weight = input("Especifique los pesos de clase (dict o 'balanced') o deje en blanco para None: ")
        if class_weight == "":
            class_weight = None
        else:
            # Intentar analizar la entrada como un diccionario
            try:
                class_weight = eval(class_weight)
            except Exception as e:
                print("Error al analizar los pesos de clase:", e)
                class_weight = None
        
        random_state = int(input("Ingrese el valor de semilla aleatoria (entero) o deje en blanco para None: ") or None)
        solver = input("Ingrese el solver a utilizar ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'): ")
        max_iter = int(input("Ingrese el número máximo de iteraciones: ") or 100)
        multi_class = input("Especifique el esquema de clasificación multinomial ('ovr', 'multinomial' o 'auto'): ")
        verbose = int(input("Ingrese el nivel de verbosidad (0 para silencioso): ") or 0)
        warm_start = input("¿Reanudar desde la solución previa? (True/False): ").lower() == 'true'
        n_jobs = int(input("Ingrese el número de trabajos paralelos (entero) o deje en blanco para None: ") or None)
        l1_ratio = input("Ingrese el valor de la proporción L1 (float) o deje en blanco para None: ")
        if l1_ratio == "":
            l1_ratio = None
        else:
            l1_ratio = float(l1_ratio)
        
        # Crear el modelo LogisticRegression con los hiperparámetros especificados por el usuario
        logistic_regression = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio
        )
        
        # Ajustar el modelo a los datos de entrada
        logistic_regression.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = logistic_regression.score(X, y)
        print("Puntaje del modelo LogisticRegression:", score)
        
        
        return logistic_regression
    
def TweedieRegressorz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "13":
        power = float(input("Ingrese el valor de power para determinar la distribución subyacente (0.0 para Normal, 1.0 para Poisson, 2.0 para Gamma, etc.): "))
        alpha = float(input("Ingrese el valor de alpha para la regularización L2 (0.0 para GLMs sin penalización, inf para penalización máxima): "))
        fit_intercept = input("¿Calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        link = input("Especifique la función de enlace ('auto', 'identity', 'log'): ")
        solver = input("Especifique el algoritmo a utilizar en el problema de optimización ('lbfgs', 'newton-cholesky'): ")
        max_iter = int(input("Ingrese el número máximo de iteraciones para el solucionador: "))
        tol = float(input("Ingrese el valor de tolerancia para la convergencia del modelo: "))
        warm_start = input("¿Reanudar desde la solución previa? (True/False): ").lower() == 'true'
        verbose = int(input("Ingrese el nivel de verbosidad (0 para silencioso): ") or 0)
        
        # Crear el modelo TweedieRegressor con los hiperparámetros especificados por el usuario
        tweedie_regressor = TweedieRegressor(
            power=power,
            alpha=alpha,
            fit_intercept=fit_intercept,
            link=link,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose
        )
        
        # Ajustar el modelo a los datos de entrada
        tweedie_regressor.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = tweedie_regressor.score(X, y)
        print("Puntaje del modelo TweedieRegressor:", score)
                
        return tweedie_regressor
    
def Perceptronz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "14":
        penalty = input("Ingrese el tipo de penalización a utilizar ('l2', 'l1', 'elasticnet') o deje en blanco para no usar penalización: ").lower() or None
        alpha = float(input("Ingrese el valor de alpha para la penalización (0.0 para desactivar la penalización): "))
        l1_ratio = float(input("Ingrese el valor de l1_ratio para la penalización elasticnet (0.0 para L2, 1.0 para L1): "))
        fit_intercept = input("¿Calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de iteraciones para el ajuste del modelo: "))
        tol = float(input("Ingrese el valor de tolerancia para la convergencia del modelo: "))
        shuffle = input("¿Mezclar los datos de entrenamiento en cada época? (True/False): ").lower() == 'true'
        verbose = int(input("Ingrese el nivel de verbosidad (0 para silencioso): ") or 0)
        eta0 = float(input("Ingrese el valor de eta0 para la tasa de aprendizaje inicial: "))
        n_jobs = int(input("Ingrese el número de trabajos para la computación paralela (deje en blanco para usar el valor predeterminado): ") or None)
        random_state = int(input("Ingrese el valor de random_state para la inicialización de los datos de entrenamiento: ") or 0)
        early_stopping = input("¿Utilizar parada temprana para detener el entrenamiento cuando el puntaje de validación no mejora? (True/False): ").lower() == 'true'
        validation_fraction = float(input("Ingrese la proporción de datos de entrenamiento para utilizar como conjunto de validación: "))
        n_iter_no_change = int(input("Ingrese el número de iteraciones sin cambio para usar como criterio de parada temprana: "))
        class_weight = input("Ingrese la configuración de pesos de clase (dict, 'balanced', None): ")
        warm_start = input("¿Reutilizar la solución del ajuste anterior como inicialización? (True/False): ").lower() == 'true'
        
        # Crear el modelo Perceptron con los hiperparámetros especificados por el usuario
        perceptron = Perceptron(
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            eta0=eta0,
            n_jobs=n_jobs,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start
        )
        
        # Ajustar el modelo a los datos de entrada
        perceptron.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = perceptron.score(X, y)
        print("Puntaje del modelo Perceptron:", score)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Coeficientes:", perceptron.coef_)
        print("Intercepción:", perceptron.intercept_)
        
        return perceptron
    
def PassiveAggressiveClassifierz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "15":
        C = float(input("Ingrese el valor de C para el paso máximo (regularización): "))
        fit_intercept = input("¿Calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de pasadas sobre los datos de entrenamiento: "))
        tol = float(input("Ingrese el valor de tolerancia para el criterio de parada: "))
        early_stopping = input("¿Utilizar parada temprana para terminar el entrenamiento cuando el puntaje de validación no mejora? (True/False): ").lower() == 'true'
        validation_fraction = float(input("Ingrese la proporción de datos de entrenamiento a reservar como conjunto de validación para la parada temprana: "))
        n_iter_no_change = int(input("Ingrese el número de iteraciones sin mejora para detener el entrenamiento si early_stopping es True: "))
        shuffle = input("¿Mezclar los datos de entrenamiento después de cada época? (True/False): ").lower() == 'true'
        verbose = int(input("Ingrese el nivel de verbosidad (0 para silencioso): ") or 0)
        loss = input("Ingrese la función de pérdida a utilizar ('hinge', 'squared_hinge'): ").lower()
        n_jobs = int(input("Ingrese el número de CPU a utilizar (deje en blanco para usar el valor predeterminado): ") or None)
        random_state = int(input("Ingrese el valor de random_state para la inicialización de los datos de entrenamiento: ") or None)
        warm_start = input("¿Reutilizar la solución del ajuste anterior como inicialización? (True/False): ").lower() == 'true'
        class_weight = input("Ingrese la configuración de pesos de clase (dict, 'balanced', None): ")
        average = input("¿Calcular los pesos promediados y almacenar el resultado en el atributo coef_? (True/False): ").lower() == 'true'
        
        # Crear el modelo Passive Aggressive con los hiperparámetros especificados por el usuario
        passive_aggressive = PassiveAggressiveClassifier(
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            loss=loss,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            average=average
        )
        
        # Ajustar el modelo a los datos de entrada
        passive_aggressive.fit(X, y)
        
        # Calcular y mostrar el puntaje del modelo
        score = passive_aggressive.score(X, y)
        print("Puntaje del modelo Passive Aggressive:", score)
                
        return passive_aggressive
    
def PassiveAggressiveRegressorz(X,y,opcion_modelo_lineal):
    if opcion_modelo_lineal == "16":
        C = float(input("Ingrese el valor de C para el paso máximo (regularización): "))
        fit_intercept = input("¿Calcular la intercepción para este modelo? (True/False): ").lower() == 'true'
        max_iter = int(input("Ingrese el número máximo de pasadas sobre los datos de entrenamiento: "))
        tol = float(input("Ingrese el valor de tolerancia para el criterio de parada: "))
        early_stopping = input("¿Utilizar parada temprana para terminar el entrenamiento cuando el puntaje de validación no mejora? (True/False): ").lower() == 'true'
        validation_fraction = float(input("Ingrese la proporción de datos de entrenamiento a reservar como conjunto de validación para la parada temprana: "))
        n_iter_no_change = int(input("Ingrese el número de iteraciones sin mejora para detener el entrenamiento si early_stopping es True: "))
        shuffle = input("¿Mezclar los datos de entrenamiento después de cada época? (True/False): ").lower() == 'true'
        verbose = int(input("Ingrese el nivel de verbosidad (0 para silencioso): ") or 0)
        loss = input("Ingrese la función de pérdida a utilizar ('epsilon_insensitive', 'squared_epsilon_insensitive'): ").lower()
        epsilon = float(input("Ingrese el valor de epsilon para la función de pérdida: "))
        random_state = int(input("Ingrese el valor de random_state para la inicialización de los datos de entrenamiento: ") or None)
        warm_start = input("¿Reutilizar la solución del ajuste anterior como inicialización? (True/False): ").lower() == 'true'
        average = input("¿Calcular los pesos promediados y almacenar el resultado en el atributo coef_? (True/False): ").lower() == 'true'
        
        # Crear el modelo Passive Aggressive Regressor con los hiperparámetros especificados por el usuario
        passive_aggressive_regressor = PassiveAggressiveRegressor(
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            loss=loss,
            epsilon=epsilon,
            random_state=random_state,
            warm_start=warm_start,
            average=average
        )
        
        # Ajustar el modelo a los datos de entrada
        passive_aggressive_regressor.fit(X, y)
        
        # También puedes imprimir otros atributos del modelo si lo deseas
        print("Coeficientes:", passive_aggressive_regressor.coef_)
        print("Intercepción:", passive_aggressive_regressor.intercept_)
        
        return passive_aggressive_regressor

    

 ################ Funcion Estimator ##################
def select_estimator():
    print("Seleccione un estimador:")
    print("1. Regresión Logística")
    print("2. Máquinas de Vectores de Soporte (SVM)")
    print("3. Árbol de Decisión")
    print("4. Bosque Aleatorio (Random Forest)")
    print("5. Gradient Boosting")
    print("6. AdaBoost")
    print("7. K Vecinos Más Cercanos (KNN)")
    print("8. Red Neuronal (MLP)")
    print("9. Naive Bayes Multinomial")
    print("10. Naive Bayes Gaussiano")
    
    option = int(input("Ingrese el número correspondiente al estimador: "))
    
    if option == 1:
        return LogisticRegression()
    elif option == 2:
        return SVC()
    elif option == 3:
        return DecisionTreeClassifier()
    elif option == 4:
        return RandomForestClassifier()
    elif option == 5:
        return GradientBoostingClassifier()
    elif option == 6:
        return AdaBoostClassifier()
    elif option == 7:
        return KNeighborsClassifier()
    elif option == 8:
        return MLPClassifier()
    elif option == 9:
        return MultinomialNB()
    elif option == 10:
        return GaussianNB()
    else:
        print("Opción no válida. Por favor, seleccione un número del 1 al 10.")
        return select_estimator()
