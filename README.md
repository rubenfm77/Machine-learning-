## Machine Learning content in R

This is my capstone project to predict employee attrition in R.

Find the files on the code folder (PowerPoint for the business department and the RMarkdown with the code). 

In the Power BI repository you'll find a pbix file with the most important variables and KPIs for the business department.

![R](https://github.com/rubenfm77/Machine-learning-/blob/main/R.jpg)

# 1. OBJETIVOS Y CONSIDERACIONES PREVIAS:

El Objetivo de este TFM es determinar la probabilidad que unos determinados empleados abandonen la empresa, pero ese análisis se podría hacer extensivo a clientes que abandonan un banco, una aseguradora etc.

El dataset original se puede localizar aquí:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  

SHRM indica que entre las razones principales por las cuales un empleado abandona su empresa se encuentran la compensación, plan de carrera, flexibilidad laboral, expectativas del empleador que considera desproporcionadas o líderes que no inspiran.

https://www.shrm.org/topics-tools/news/report-hr-pros-rank-top-reasons-turnover

Se considera que un índice óptimo de abandono por parte de los empleados se debería situar por debajo del 10%:

https://www.hirequotient.com/hr-glossary/what-is-attrition-rate

En este dataset de IBM nuestra variable dependiente será "Attrition".

Como decíamos, nuestro análisis se centrará en determinar qué proporción de empleados abandonan la empresa y si existe un patrón por centros, edad, sexo o determinadas funciones. El objetivo, con estos modelos predictivos, es no solamente retener al talento adoptando medidas tales como planes de carrera, formación, promoción, aumento de salario etc, sino también permitir a la empresa prepararse mejor para el futuro, sabiendo qué vacantes va a necesitar cubrir, tanto por jubilación preparando un relevo generacional ordenado con traspaso del conocimiento, como las vacantes que dejarán libres determinados perfiles que no se podrá/interesará retener.

Finalmente, probaremos diversos modelos, principalmente de H2O y determinaremos el % de acierto de cada uno de ellos para acabar seleccionando aquel que nos de un AUC (Area Under Curve) más elevado. 

Llegados a este punto dónde mecionamos el AUC, hay que también hacer mención a ROC (Receiver Operating Characteristic). ROC es una curva que se utiliza para encontrar la eficacia de un clasificador, siendo ROC una curva de 2-D, dónde el eje X representa el FPR (False Positive Rate) y el eje Y representa el TPR (True Positive Rate). En este caso, ROC se cuantifica como AUC viendo qué área cubre la curva ROC. Un clasificador perfecto tendría un AUC de 1.0, si bien esa cifra es difícil de alcanzar en el mundo real y un rango entre 0.6 y 0.9 se consideraría un buen clasificador.

Empezaremos cargando el CSV y realizando un análisis exploratorio de los datos de nuestro dataset.


Cargamos algunas librerías:

```{r}
suppressPackageStartupMessages({
library(dplyr)
library(data.table)
library(tidytable)
library(janitor)
library(stringr)
library(magrittr)
library(ggplot2)
library(gapminder)
library(ggrepel)
library(plotly)
library(tidyverse)
library (treemapify)
library (treemap)
library(GGally)
library(car)
library(tidyr)
library(h2o)
library(rattle)
library(rpart.plot)
library(caret)
library(randomForest) 
library (ROSE)
library (rsample)
library(caret)
library(vtreat)
library (inspectdf)
library (lares)
library(skimr)
})
```


# 2. DATASET IBM

## 2.1 ANÁLISIS EXPLORATORIO Y TRANSFORMACIONES:

Nuevamente, nuestro objetivo va a ser analizar las distintas variables del dataset e intentar ver aquellas que tienen más peso a la hora que un empleado abandone la empresa. Nuestro modelo será capaz de predecir qué empleados abandonarán la empresa.

El dataset que podemos localizar aquí:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset 


Empezamos cargando el fichero:

```{r}
IBM <- fread("C:\\ruben_fernandez\\IBM.csv")
```

Realizamos un summary de los datos y a primera vista, ya vemos que es un dataset mucho más rico, con variables como la distancia que recorre el empleado, nivel de educación, número de horas trabajadas, si hace horas extras, los años desde la última promoción etc:

```{r}
summary(IBM)
```


Vemos que tenemos 1470 filas y 35 variables, lo cual confirma que es un dataset con más información en lo que se refiere a número de variables, pero menos filas que nuestro anterior dataset. El tipo de variables que hay en el dataset nos debería permitir, como ya hemos dicho, determinar la probabilidad que un empleado abandone la empresa, poniendo foco en las variables que inciden más en ese hecho y tomar decisiones y acciones basadas en datos que se traduzcan en la retención del talento antes que éste deje la empresa:

```{r}
dim(IBM)
```


Vamos a ver más detalles del dataset com skimr y vemos ya que, seguramente, algunas variables por su uniformidad, probablemente no las descartemos para nuestro modelo:

```{r}
skim(IBM)
```

Podríamos calcular que % de empleados dejan nuestra empresa por departamento y vemos que el % más alto se sitúa en el departamento de ventas, como se podía presuponer, por la presión comercial, seguido de RRHH y R&D, aunque como veremos, en número absoluto es R&D el departamento con más bajas:


```{r}
status_count<- as.data.frame.matrix(IBM %>%
group_by(Department) %>%
select(Department, Attrition) %>%

ungroup(Department) %>% 
table())

status_count <- status_count %>% 
  mutate(total = No + Yes,
         turnover_rate = (Yes/(total + No)/2)*100)
status_count %>% 
  as.data.frame() %>% 
  select(turnover_rate) %>% 
  arrange(desc(turnover_rate))
```

```{r}
mean(status_count$turnover_rate)

range(status_count$turnover_rate)
```

```{r}  
glimpse(IBM)
```

Dado que nuestra variable objetivo o dependiente será Attrition y ésta es categórica, la vamos a convertir a booleano con 0 y 1, siendo 1 Yes y 0 No, para lo cual realizamos estos pasos:

```{r}
IBM$Attrition[IBM$Attrition=="Yes"]=1
IBM$Attrition[IBM$Attrition=="No"]=0
IBM$Attrition=as.numeric(IBM$Attrition)

summary(IBM)
dim(IBM)

```

Si calculamos el % de abandonos, vemos que nos situamos por encima del 10% que sería el máximo deseable:

```{r}
IBM %>%
summarize(avg_turnover_rate = mean(Attrition))
```

## 2.2 PLOTS Y ANÁLISIS DE LAS DISTINTAS VARIABLES:

Finalmente, a la vista de los inconvenientes que presentó nuestro anterior dataset, pensamos que será conveniente convertir a factor el resto de variables, pero empezaremos por realizar, además del análisis exploratorio de los datos, el ploteo de diversas variables.

De entrada, podemos observar que, al contrario del anterior dataset, en este los hombres abandonan más la empresa que las mujeres y nos sorprende que haya un mayor abandono en el caso del departamento de R&D que de ventas en números absolutos.


```{r}
Sexos<- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(Gender),fill = as.factor(Department)),data=Sexos,position = position_stack())
```

Repetimos el ejercicio anterior sin filtros para poder ver la distribución de la plantilla por sexos y vemos que hay un mayor número de hombres que de mujeres, cosa que explica la diferencia que veíamos anteriormente y no se observa, por lo tanto, un mayor abandono de la empresa solamente por razones de sexo, ya que hay 294 hombres más que mujeres y es lógico que, en números absolutos haya más bajas:

```{r}
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(Gender),fill = as.factor(Department)),data=IBM,position = position_stack())
```

También podríamos simplemente realizar un conteo y vemos que el número de hombres es un 0.5% superior al de mujeres:

```{r}
IBM %>%
  count(Gender)
```

Sospechamos que Attrition será una variable con un desbalanceo importante y lo confirmamos:

```{r}
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(Attrition),fill = as.factor(Department)),data=IBM,position = position_stack())
```

Nuevamente podríamos realizar un conteo y vemos que 237 empleados han abandonado la empresa:

```{r}
IBM %>%
  count(Attrition)
```

Vamos a revisar diversas variables como ya hicimos en el anterior dataset. En el caso de mayoría de edad, no parece que aporte valor y, probablemente, la eliminaremos:

```{r}
categoricalIBM <- inspect_cat(IBM)
show_plot(categoricalIBM)
```

Abundan las variables numéricas:

```{r}
numericalIBM <- inspect_num(IBM)
show_plot(numericalIBM)
```

Aquí se puede ver con más detalle:

```{r}
typesIBM <- inspect_types(IBM)
show_plot(typesIBM)
```

Vemos que hay variables con una correlación alta, pero este gráfico no nos permite verlo en detalle por la gran cantidad de variables:

```{r}
corrIBM <- inspect_cor(IBM)
show_plot(corrIBM)
```

Vamos a realizar un ejercicio similar con lares y las 25 variables más relevantes y esto nos da información no definitiva de por dónde podríamos buscar relaciones entre variables:

```{r}
corr_cross(IBM, max_pvalue= 0.05, top=25, grid = T)
```

Se trata de otro dataset sin NA:

```{r}
nulosIBM <- inspect_na(IBM)
show_plot(nulosIBM)
```

Nuevamente nos llama la atención la variable de mayoría de edad:

```{r}
balanceIBM <- inspect_imb(IBM)
show_plot(balanceIBM)
```



Pensamos que determinadas funciones deben tener más tendencia a dejar la empresa, por lo que vamos a comprobar si es así a la vista que el departamento de R&D parece que es el más perjudicado. Este gráfico nos da una información muy valiosa porqué vemos que los técnicos de laboratorio, ejecutivos de ventas, científicos de investigación y representantes de ventas, son, por este orden, las figuras con más propensión a dejar la empresa, aunque ya vimos al principico que en % el departamento de ventas es el que registra un mayor número de abandonos, aunque no sea así en números absolutos:


```{r}
Job<- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(JobRole),fill = as.factor(Department)),data=Job,position = position_stack())+
 theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
```

Puede la formación estar relacionada con nuestra variable Attrition, es decir, ¿una persona más formada o con una formación en un campo específico, tiene más riesgo de dejar la empresa? Veámoslo en este gráfico, dónde vemos que unos campos destacan más que otros:

```{r}
Job<- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(EducationField),fill = as.factor(Department)),data=Job,position = position_stack())+
 theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
```

¿Y el estado civil? Pues parece que las personas divorciadas son las que tienen una menor tendencia a dejar la empresa y en la parte opuesta se encuentran los solteros/as:

```{r}
Job<- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(MaritalStatus),fill = as.factor(Department)),data=Job,position = position_stack())+
 theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
```

¿Viajar frecuentemente puede hacer que los empleados/as dejen la empresa? Pues aquellos perfiles que viajan rara vez son los que más dejan la empresa y en la parte opuesta están los que no viajan. Necesitamos algo más de concreción sobre las personas que viajan y tal vez un ratio nos dará más información que el gráfico, aunque ya vemos que la dejan más que las personas que no viajan:

```{r}
Job<- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x =as.factor(BusinessTravel),fill = as.factor(Department)),data=Job,position = position_stack())+
 theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
```   


Creamos una nueva columna que se llamará Abandono:

```{r}
IBM <- transform(IBM, Abandono =ifelse (Attrition == 1, "YES", "NO"))
dim(IBM)
```


Realizamos el ratio y vemos que, efectivamente, en % aquellos que viajan frecuentemente son los que más abandonan la empresa:

```{r}
travel_count<- as.data.frame.matrix(IBM %>%
group_by(BusinessTravel) %>%
select(BusinessTravel, Abandono) %>%

ungroup(BusinessTravel) %>% 
table())

travel_count <- travel_count %>% 
  mutate(total = NO + YES,
         RatioAbandonoTravel = (YES/(total + NO)/2)*100)
travel_count %>% 
  as.data.frame() %>% 
  select(RatioAbandonoTravel) %>% 
  arrange(desc(RatioAbandonoTravel))
``` 

Pensamos que la distancia recorrida hasta el centro de trabajo podría ser uno de los motivos por los cuales un trabajador abandona la empresa, pero no parece que sea éste el motivo principal y  es tal vez en las distancias más cortas, cuando hay un mayor nivel de 1 en la variable Attrition. Seguramente la empresa contempla medidas como el teletrabajo, pero hay cierto incremento de abandonos en los empleados que recorren entre 20-25 millas para ir descendiendo posteriormente.

```{r}
featurePlot(x=IBM[,6],y=as.factor(IBM$Attrition),plot="density",auto.key = list(columns = 2))
```

De todos modos, tal vez un boxplot nos arroje algo de detalle y parece que, de media, la distancia recorrida es algo superior para los empleados que dejan la empresa y esta variable tiene un cierto peso, tal y como ya se deja entrever en el featurePlot:

```{r}
ggplot(IBM, aes(x = as.factor(Attrition), y = DistanceFromHome)) + geom_boxplot()
```

Podemos intentar ver la relación entre estas variables de Distancia y el salario percibido, estando determinado el color por el sexo y el tamaño por los años trabajados, pero no se ve una relación clara:


```{r}
IBM %>% 
  filter(Attrition==1) %>%
  ggplot(aes(DistanceFromHome, MonthlyIncome)) + 
  geom_point(aes(color = Gender,
                 size = TotalWorkingYears))+
  geom_smooth()+
  labs(x = "Distancia",
       y = "Ingresos",
       title = "Distancia vs salario")+
  theme_minimal()
```


Por otro lado, vemos que aquellos que perciben un salario por hora más bajo, tienen más tendencia a dejar la empresa:

```{r}
featurePlot(x=IBM[,13],y=as.factor(IBM$Attrition),plot="density",auto.key = list(columns = 2))
```

Repitamos el ejercicio con otro boxplot, pero se confirma lo que ya vimos, es decir, que hay una ligera subida para la categoría 1 de la variable Attrition si el salario percibido por hora es más bajo:

```{r}
ggplot(IBM, aes(x = as.factor(Attrition), y = HourlyRate)) + geom_boxplot()
```

Veamos qué sucede con la variable OverTime (horas extras). Se observa un ligero repunte en el caso de aquellos empleados que realizan horas extras, pero no parece que haya una diferencia significativa, excepto en el departamento de R&D. Habrá que crear un ratio para poder hacer un zoom más claro en el siguiente punto.

```{r}
OverTime <- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x = as.factor(OverTime),fill = as.factor(Department)),data=OverTime,position = position_stack())+
 theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
```   


Con ratio tenemos más detalle y vemos que abandonan la empresa en un % bastante más alto que aquellos que hacen horas extras frente a los que no las hacen:

```{r}
overtime_count<- as.data.frame.matrix(IBM %>%
group_by(OverTime) %>%
select(OverTime, Abandono) %>%

ungroup(OverTime) %>% 
table())

overtime_count <- overtime_count %>% 
  mutate(total = NO + YES,
         RatioAbandonoHoras = (YES/(total + NO)/2)*100)
overtime_count %>% 
  as.data.frame() %>% 
  select(RatioAbandonoHoras) %>% 
  arrange(desc(RatioAbandonoHoras))
``` 


Vamos a intentar analizar OverTime un poco más y a establecer unos tramos de salario. Parece que sí hay cierto incremento de la variable Attrition 1 en salarios bajos que, en general, aumenta con el incremento de las horas que se trabajan:

```{r}
IBM %>%
  mutate(SalaryRange = case_when(MonthlyIncome < 2500 ~ 'low',
                           MonthlyIncome < 5500 ~ 'med',
                           MonthlyIncome > 5500 ~ 'high'))%>%
  ggplot(aes(x = as.factor(Attrition), y = as.factor(StandardHours), colour = StandardHours)) +
  geom_boxplot(outlier.colour = NA) + 
  geom_jitter(alpha = 0.05, width = 0.1) +
  facet_wrap(vars(SalaryRange), 
             scales = "free", 
             ncol = 3) +
  xlab("Abandono Empleado") +
  ylab("Horas Trabajadas al Mes")

```  

Una variable interesante es NumCompaniesWorked, probablemente, un trabajador que ha pasado por varias empresas, tendrá una mayor tendencia a abandonar IBM ya que está más acostumbrado al cambio, pero tampoco parece que éste sea el caso. Son aquellos empleados que se sitúan en la franja de 1 empleo previo los que abandonan con mucha diferencia la empresa, especialmente R&D y ventas. Pensamos que valdría la pena realizar acciones de clima laboral antes del primer año en esos perfiles que llevan poco tiempo en la empresa en forma de acompañamiento, formación, team building etc:

```{r}
Companies <- as.data.frame(IBM %>%
filter(Attrition==1))
ggplot() + geom_bar(aes(y = ..count..,x = as.factor(NumCompaniesWorked),fill = as.factor(Department)),data=Companies,position = position_stack())+
 theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
```   

Vamos a intentar ver si existe relación entre las variables de PerformanceRating, RelationshipSatisfaction y HourlyRate, pero parece que la relación entre estas variables es débil:

```{r}
IBM %>% 
  select(PerformanceRating, RelationshipSatisfaction, HourlyRate) %>% 
  cor() %>% 
  corrplot::corrplot(method = "number")
```   

Es posible que un empleado abandone la empresa porqué se siente valorado injustamente. Pues parece que es el caso, aunque también podría ser una política de la empresa para intentar prescindir de aquellos empleados con un rendimiento inferior. En general, un empleado de cualquier rango salarial con valoración +- <3.5 tienes más posibilidades de dejar la empresa:

```{r}
IBM %>%
  mutate(SalaryRange = case_when(MonthlyIncome < 2500 ~ 'low',
                           MonthlyIncome < 5500 ~ 'med',
                           MonthlyIncome > 5500 ~ 'high'))%>%
  ggplot(aes(x = as.factor(Attrition), y = as.factor(PerformanceRating),           colour=PerformanceRating)) +
  geom_boxplot(outlier.colour = NA) + 
  geom_jitter(alpha = 0.05, width = 0.1) +
  facet_wrap(vars(SalaryRange), 
             scales = "free", 
             ncol = 3) +
  xlab("Abandono Empleado") +
  ylab("Rating Empleado") 
```   

Deberíamos también preguntarnos si aquellos empleados que llevan mucho tiempo en la función actual tienen tendencia a dejar la empresa, si no han promocionado. Vemos que sí existe esta tendencia para aquellos empleados que llevan menos años en la empresa y en la función, cosa que enlaza con lo que ya habíamos visto y es que este tipo de empleados, tienen mayor tendencia en general a dejar la empresa, tal vez buscando promociones en otras al ser perfiles, probablemente, más junior:

```{r}
IBM %>% 
  ggplot(aes(x = as.factor(Attrition), y = as.factor(YearsInCurrentRole), colour = YearsInCurrentRole)) +
  geom_boxplot(outlier.colour = NA) + 
  geom_jitter(alpha = 0.05, width = 0.1) +
  facet_grid(cols = vars(YearsSinceLastPromotion))+
  xlab("Abandono Empleado") +
  ylab("Años en la función") 
```   

Vamos a intentar ver si el salario y el equilibrio entre la vida laboral y la personal afectan al abandono de la empresa, teniendo en cuenta que 1 es el valor más bajo y 4 el más alto. Parece que hay cierta influencia para aquellos que se sitúan por debajo del 3.5 aproximadamente:

```{r}
IBM %>% 
  mutate(SalaryRange = case_when(MonthlyIncome < 2500 ~ 'low',
                           MonthlyIncome < 5500 ~ 'med',
                           MonthlyIncome > 5500 ~ 'high'))%>%
  ggplot(aes(x = as.factor(Attrition), y = as.factor(WorkLifeBalance),           colour=WorkLifeBalance)) +
  geom_boxplot(outlier.colour = NA) + 
  geom_jitter(alpha = 0.05, width = 0.1) +
  facet_wrap(vars(SalaryRange), 
             scales = "free", 
             ncol = 3) +
  xlab("Abandono Empleado") +
  ylab("Equilibrio Trabajo-Familia") 
```   

La compensación, según SHRM, es una de las razones principales por las que un empleado abandona su empresa y ya hemos visto la importancia de dicha variable, por lo que vamos a crear dos nuevas columnas, dónde establecemos tres tramos similares a los que hemos establecido antes creando un ratio dividiendo el salario entre el salario medio:

```{r}
IBM[ , `:=`(salario_medio = 
                median(MonthlyIncome)),by = .(JobLevel) ]
IBM[ , `:=`(RatioSueldo =     (MonthlyIncome/salario_medio)), by =. (JobLevel)]
IBM[ , `:=`(NivelSalarial = 
               factor(fcase(RatioSueldo
                            %between% list(0.75,1.25), "Medio",
                            RatioSueldo 
                            %between%  list(0,0.74), "Bajo",
                            RatioSueldo
                            %between% list(1.26,2),  "Alto"),
                            levels = c("Bajo","Medio","Alto"))),
                            by = .(JobLevel) ]

```   

```{r}
dim(IBM)
```

Vamos a ver si hay más empleados satisfechos que los que no lo están y vemos que la mayoría lo están (por encima del 60% tienen putuación 3 o 4):


```{r}
IBM[, list(conteo = .N, rate = (.N/nrow(IBM))), by = JobSatisfaction]
```


Pensamos que, según hemos visto, a mayor satisfacción del empleado, menos probabilidad que abandone la empresa, en la línea de SHRM y vemos que, en líneas generales, los empleados que dejan la empresa, en general están menos satisfechos. 

Omitimos acentos e interrogantes ya que nos generan error:

```{r}

theme_custom <- function(){ 
  
  theme(
    
       strip.background = element_rect(colour = "black", fill = "lightgrey"),
    
     axis.title.x = element_blank(),               
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),                
    
    legend.box.background = element_rect())
  
}


library(RColorBrewer)
myCol <- rbind(brewer.pal(8, "Blues")[c(5,7,8)],
               brewer.pal(8, "Reds")[c(4,6,8)])




plot_jobsatisfaction <- ggplot(IBM, aes(x = Abandono, y = JobSatisfaction,fill = Abandono))+
                          geom_boxplot(width=0.1)+
                          scale_fill_manual(values = myCol)+
                          ylab("Satisfaccion Empleado")+
                          xlab("Abandono Empleado")+
                          theme_custom()+
                          ggtitle("Los Empleados que dejan la empresa estan menos satisfechos?")
plot_jobsatisfaction

```   

Si se combina un salario bajo y un empleado que no está satisfecho, pensamos que es más probable que deje la empresa y con este plot se confirma que, en general, los empleados que dejan la empresa tienen un salario más bajo:

```{r}
Plot_Ingresos <- ggplot(IBM, aes(x = as.factor(JobSatisfaction), y = MonthlyIncome,fill = Abandono))+
                       geom_boxplot()+
                       scale_fill_manual(values = myCol)+
                       ylab("Salario")+
                       xlab("Satisfaccion Empleado")+
                       theme_custom()+
                       ggtitle("Ingresos y satisfaccion empleado vs sueldo")
               
Plot_Ingresos
```   

Cómo ya hicimos, vamos a intentar ver si las variables de clima laboral afectan a la decisión del empleado de abandonar la empresa, aunque parece que ya hemos ido viendo que así es y este boxplot lo confirma de manera clara con los trabajadores que dejan la empresa menos satisfechos que los que permanecen en ella y, además, tienen un sueldo más bajo:

```{r}
plot_CompRatio <- ggplot(IBM, 
                      aes(x = as.factor(JobSatisfaction),y = RatioSueldo,fill = Abandono))+
                      geom_boxplot()+
                      scale_fill_manual(values = myCol)+
                      ylab("Ratio Salarial")+
                      xlab("Satisfaccion")+
                      theme_custom()+
                     ggtitle("Relacion Salario y Satisfaccion con Abandono")
                 
plot_CompRatio
                          
```  

Vamos a crear una serie de gráficos que nos permitan ver la relación entre variables. Mirando a ambos lados de los rectángulos y los colores podemos ver que el area del rectángulo representa la proporción de casos para cualquier combinación de niveles, mientras que el color indica el grado de relación entre variables, cuanto más se desvía el color del gris, más cuestionable es la independencia estadística entre las combinaciones (que se representan en la escala de Pearson). En general el azul oscuro supone más casos que los esperados por una ocurrencia casual, mientras que el rojo oscuro representa menos casos de los esperados.

```{r}
library(vcd)
library(vcdExtra)

mosaic(~ Abandono + EnvironmentSatisfaction, data = IBM,
       main = "Clima laboral vs Abandono", shade = TRUE, legend = TRUE)
mosaic(~ Abandono + JobInvolvement, data = IBM,
       main = "Implicacion vs Abandono", shade = TRUE, legend = TRUE)
mosaic(~ Abandono + WorkLifeBalance, data = IBM,
       main = "Equilibrio vida personal-laboral vs Abandono", shade = TRUE, legend = TRUE)
mosaic(~ Abandono + RelationshipSatisfaction, data = IBM,
       main = "Buena relacion vs Abandono", shade = TRUE, legend = TRUE)
```         

Empezaremos por mirar correlaciones para, de este modo, poder deshacernos de aquellas variables que, pensamos, no aportarán valor identificando las variables numéricas y deshechando aquellas que tengan una correlación que supere el 0.5, tomando una muestra del dataset:


```{r}
IBMClean <- IBM[sample(.N, 500)]
IBMClean <- IBMClean[,-c("DailyRate","EducationField",  "EmployeeCount","EmployeeNumber","MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel","Gender","Over18", "OverTime", "salario_medio", "Attrition")]

IBMReducido <-as.data.frame(unclass(IBMClean),stringsAsFactors=TRUE)
```


```{r}
nums <- unlist(lapply(IBMReducido, is.numeric))

ibm_nums <- IBMReducido[,nums]

head(ibm_nums)

correlationMatrix <- cor(ibm_nums)

correlationMatrix

Altacorrelacion <- findCorrelation(correlationMatrix, cutoff=0.5)

colnames(ibm_nums[,Altacorrelacion])
correlationMatrix[,Altacorrelacion]
```

Vemos que algunas variables como edad y nivel salarial tienen una alta correlación, la valoración (PerformanceRating) y el incremento salarial (PercentSalaryHike).


## 2.3 PRUEBA DE DISTINTOS MODELOS DE H20

Como ya vimos en en el anterior dataset, H2O daba unos muy buenos resultados al probar distintos algoritmos por nosotros, de manera que esta será nuestra primera opción. Eliminaremos aquellas variables que, pensamos, no son importantes para nuestro modelo y normalizaremos los datos ya que MonthlyRate, por ejemplo, tiene unos valores muy altos que, tal vez, perjudicaría al modelo:

Eliminamos columnas que no aportan valor como la mayoría de edad, el conteo de empleados etc:

```{r}
IBM$EducationField <-NULL
IBM$EmployeeCount <- NULL
IBM$EmployeeNumber <- NULL
IBM$Over18 <- NULL
IBM$StandardHours <- NULL
```

Nos aseguramos con la función que los valores más alejados, no perjudiquen el modelo aplicando la misma a diversas variables:

```{r}
Regulariza <- function(x)
{
  return ((x-min(x))/ (max(x)-min(x)))
}

IBM$DailyRate <- Regulariza(IBM$DailyRate)
IBM$DistanceFromHome <-Regulariza(IBM$DistanceFromHome)
IBM$HourlyRate <-Regulariza(IBM$HourlyRate)
IBM$MonthlyIncome <-Regulariza(IBM$MonthlyIncome)
IBM$MonthlyRate <-Regulariza(IBM$MonthlyRate)
IBM$NumCompaniesWorked <-Regulariza(IBM$NumCompaniesWorked)
IBM$PercentSalaryHike <-Regulariza(IBM$PercentSalaryHike)
IBM$TotalWorkingYears <-Regulariza(IBM$TotalWorkingYears)
IBM$YearsAtCompany <-Regulariza(IBM$YearsAtCompany)
IBM$YearsInCurrentRole <-Regulariza(IBM$YearsInCurrentRole)
IBM$YearsSinceLastPromotion <-Regulariza(IBM$YearsSinceLastPromotion)
IBM$YearsWithCurrManager <-Regulariza(IBM$YearsWithCurrManager)
```

Deberíamos abordar también algunas variables categóricas para que nuestro modelo las pueda interpretar y la variable Attrition:

```{r}
IBM$Gender <- as.factor(IBM$Gender)
IBM$BusinessTravel <- as.factor(IBM$BusinessTravel)
IBM$Department <- as.factor(IBM$Department)
IBM$Education <- as.factor(IBM$Education)
IBM$EnvironmentSatisfaction <- as.factor(IBM$EnvironmentSatisfaction)
IBM$JobInvolvement <- as.factor(IBM$JobInvolvement)
IBM$Attrition <- as.factor(IBM$Attrition)
IBM$JobSatisfaction <- as.factor(IBM$JobSatisfaction)
IBM$JobRole <- as.factor(IBM$JobRole)
IBM$MaritalStatus <- as.factor(IBM$MaritalStatus)
IBM$OverTime <- as.factor(IBM$OverTime)
IBM$PerformanceRating <- as.factor(IBM$PerformanceRating)
IBM$RelationshipSatisfaction <- as.factor(IBM$RelationshipSatisfaction)
IBM$StockOptionLevel <- as.factor(IBM$StockOptionLevel)
IBM$WorkLifeBalance <- as.factor(IBM$WorkLifeBalance)
```

Preparamos nuestro modelo balanceando las clases porque ya vimos el desbalanceo existente en la variable Attrition y tratamos nuestro dataset para que H2O lo pueda procesar:

```{r}
localH2O = h2o.init(nthreads = 1)
IBMh2o <- as.h2o(IBM)
Y <- "Attrition"
X <- setdiff(names(IBM), Y)
```

Ponemos en marcha nuestro modelo y vemos que el mejor resultado con un AUC del 84% y un AUCPR del 65%. Era de esperar, como ya vimos, puesto que los modelos StackedEnsemble son una clase de algoritmos que seleccionan las mejores partes de los distintos modelos para darnos un modelo a nuestra medida:

```{r}
ModeloIBM <- h2o.automl(
  y = Y,
  x = X,
  training_frame = IBMh2o,
  max_runtime_secs = 600,
  balance_classes = TRUE,
  seed = 1)
ModeloIBM
```

Analizamos el comportamiento de nuestro modelo para lo cual invocamos leaderboard:

```{r}
Analisis <- ModeloIBM@leaderboard
print(Analisis)
ModeloIBM@leader
```

Como en nuestro anterior dataset, podemos realizar diversas pruebas con otros modelos, para ver si H2o con AutoML nos ha devuelto el mejor modelo posible o existen otros que nos den un resultado similar. Vamos a utilizar algunos de los modelos que ya vimos.

RandomForest nos da un AUC del 77%, pero es peor que nuestro primer modelo con AutoML:

```{r}
ModeloIBMForest <- h2o.randomForest(
  y = Y,
  x = X,
  training_frame = IBMh2o,
  max_runtime_secs = 200,
  nfolds = 5,
  balance_classes = TRUE,
  seed = 1)
ModeloIBMForest
```

Ploteamos y vemos que nuevamente Overtime es una variable con mucha importancia para el modelo junto con el JobRole (función) y los ingresos:

```{r}
h2o.varimp_plot(ModeloIBMForest)
```

Gradient Boosting Machine es ligeramente mejor que RandomForest y tiene un 78% de AUC:

```{r}
ModeloIBMGBM <- h2o.gbm(
  y = Y,
  x = X,
  training_frame = IBMh2o,
  max_runtime_secs = 200,
  nfolds = 5,
  balance_classes = TRUE,
  seed = 1)
ModeloIBMGBM
```

Podemos nuevamente plotear las variables más importantes para este modelo GBM.Para este modelo es más importante la variable JobRole (función) que Overtime (horas extras), si bien, ambas tienen bastante peso:

```{r}
h2o.varimp_plot(ModeloIBMGBM)
```

Vamos a utilizar el algoritmo deeplearning que tiene un resultado del 82% y parece que trabaja mejor con este dataset de variables numéricas que con las categóricas:

```{r}
ModeloIBMDL <- h2o.deeplearning(
  y = Y,
  x = X,
  training_frame = IBMh2o,
  max_runtime_secs = 200,
  nfolds = 5,
  balance_classes = TRUE,
  seed = 1)
ModeloIBMDL
```

Repetimos el ejercicio de plotear las variables más importantes para el modelo deeplearning y vemos que da un peso muy similar a todas ellas, siendo la primera WorkLifeBalance 2 que se consideraba como buen equilibrio entre ambas:

```{r}
h2o.varimp_plot(ModeloIBMDL)
```


# 3. CONCLUSIONES:

De 1233 empleados, vemos que 237 han abandonado la empresa, siendo ventas con un 5,75% de ratio el que más abandonos registra, seguido de RRHH con un 5,26% y R&D con un 3,72%. La media global de abandonos es del 16% y en números absolutos es R&D el departamento que más bajas registra.

Por sexos los hombres tienen más tendencia a dejar la empresa, pero también es cierto que en la plantilla el número de hombres es superior al de mujeres (882 hombres y 588 mujeres).

Por funciones, los técnicos de laboratorio, seguidos de ejecutivos de ventas, científicos de investigación y representantes de ventas por este orden, son quienes tienen más propensión a "Attrition".

Por su formación, las personas con formación en ciencias de la naturaleza, médicas y de marketing, también en este orden, son quienes tienen más riesgo de dejar la empresa.

En lo referente al estado civil, los solteros son los que más abandonan la compañía, seguidos de casados y el último lugar separados.

Las personas que viajan frecuentemente dejan más la empresa que aquellos que no lo hacen, aunque en números absolutos la dejan más aquellos que no viajan. Esta es una variable con bastante importancia.

La distancia recorrida hasta el centro de trabajo vemos que tiene también cierta relación con el abandono por parte de los empleados, especialmente el tramo de 20 a 25 millas, para ir decreciendo progresivamente. De media, los empleados que recorren más millas dejan más la empresa.

Los empleados que dejan más la empresa, son aquellos del tramo entre 0 y un 1 año, especialmente los de este segundo tramo.

En lo referente a horas trabajadas, también dejan más la empresa los trabajadores que están 40 horas o más trabajando a la semana.

En lo que respecta a las horas extraordinarias, estos trabajadores también dejan más la empresa, especialmente los del departamento R&D. Al igual que viajar, esta variable pesa a la hora de dejar la compañía.

El salario, como indica SRHM, es una de las principales causas por las cuales se deja el empleo y si ponemos el foco en esta variable conjugada con otras vemos que los empleados que trabajan más horas y tienen un salario del rango que hemos determinado como bajo (<2500), dejan más la empresa. Lo mismo sucede con aquellas personas en el rango salarial más bajo y con unos niveles de conciliación entre la vida laboral y familiar de 3,5 o inferior.

También en la línea de lo que indica SRHM, los empleados que se sienten peor valorados, suelen dejar la empresa y vemos que es lo que refleja este dataset de IBM dónde los empleados que se sienten peor valorados suelen dejar la empresa más con independencia de su rango salarial, con lo cual podemos concluir que esa variable puede llegar a pesar incluso más que la compensación.

Por lo que respecta a los modelos, hemos utilizado AutoML y los testeamos contra RandomForest, GBM y DeepLearning, eliminando aquellas variables que no se han considerado tan relevantes como ser mayor de edad, id de empleado etc. 

Se ha aplicado una función para minimizar el efecto de los outliers y de ese modo poder asegurar la mayor eficacia posible del modelo y se han balanceado las clases de la variable Attrition que, como vimos, estaba desbalanceada.

Con este dataset, hemos podido verificar las causas por las cuales un empleado deja su empresa y así lo evidencian los distintos modelos. Por ejemplo, para RandomForest las horas extras, la función, los ingresos, la edad o las horas trabajadas tienen relevancia a la hora que un empleado deje la empresa. Para GBM son la función, las horas extras, la edad, los ingresos, clima laboral o la satisfacción en el trabajo las que más pesan en la decisión de dejar una empresa. Finalmente, para el modelo de DeepLearning que se ha revelado como bastante efectivo, entiende que clima laboral, equilibrio entre vida personal y familiar o no viajar frecuentemente, entre otras, son variables que van a tener mucho peso a la hora que un empleado deje la empresa en la que presta sus servicios. Es decir, los distintos modelos están bastante alineados en lo referente a las variables más importantes.

Hemos visto que, en buena medida, se confirma lo que indica SHRM y es que variables como la compensación, la distancia recorrida, la valoración del empleado y el clima laboral pueden precipitar el hecho que un empleado abandone la empresa. En el caso de este dataset las horas extraordinarias y viajar, serían dos de las variables con más peso, pero combinadas con otras como el salario etc. tienen un efecto multiplicador en el desistimiento por parte del empleado.

Este tipo de modelos nos permiten identificar en nuestras organizaciones los perfiles que tenemos catalogados como talento y que tienen riesgo de abandonar la empresa por los motivos expuestos, de manera que podemos accionar diversas palancas en plazos de tiempo anteriores a los que se producen las bajas, bien sea estableciendo un aumento de la compensación que se percibe, un plan de carrera, acciones de team building o coaching para los mánagers etc. Este tipo de decisiones proactivas, suponen no solamente una mejora del clima laboral y de la percepción que el empleado tiene de su empleador, sino también un ahorro económico para la empresa y que ésta se convierta en un "great place to work", mejorando el estatus de la misma en el mercado laboral y que sea más fácil captar talento.

Estos modelos, como ya dijimos para el modelo del anterior dataset, deben ser monitorizados y seguir alimentándose con los datos que genere la organización, para que, no solamente el modelo pueda seguir aprendiendo en base a éstos, sino también para realizar modificaciones en el mismo con los cambios en la organización, composición de la plantilla, objetivos de negocio etc.

Y, como ya dijimos al principio, este tipo de modelos se pueden hacer extensivos a otros campos distintos al de RRHH, como por ejemplo pueden ser la pérdida de clientes en todo tipo de empresas (bancos, aseguradores, empresas de telefonía etc). Nos permiten ser proactivos y tener más claras las decisiones y acciones que se deben tomar, identificando los perfiles con más riesgo de dejar la empresa.

# 4. ELABORACION DE CUADRO DE MANDO EN PBI EN BASE A LAS CONCLUSIONES DE MACHINE LEARNING

En base a las conclusiones extraídas que nos han permitido determinas las variables con un mayor peso en el desistimiento por parte de los empleados, construímos un cuadro de mando que permita a las unidades de negocio (RRHH) controlar de forma sencilla las mismas y tomar medidas correctoras, si fuese necesario:

![HR1](https://github.com/rubenfm77/POWER-BI/blob/main/HRDash_1.jpg)
![HR2](https://github.com/rubenfm77/POWER-BI/blob/main/HRDash_2.jpg)

# 5. BIBLIOGRAFÍA Y RECURSOS

Para la elaboración de este TFM, se ha utilizado el material proporcionado en los módulos así como estas referencias:

  https://www.aihr.com/blog/tutorial-people-analytics-r-employee-churn/


  https://github.com/martins-jean/Employee-turnover-prediction-in-R/blob/main/turnover_prediction_in_R.ipynb
  
  https://towardsdatascience.com/human-resource-analytics-can-we-predict-employee-turnover-with-caret-in-r-3d871217e708

- Modelos machine learning: https://docs.h2o.ai/, https://www.aihr.com/blog/tutorial-people-analytics-r-employee-churn/ https://www.youtube.com/watch?v=i-xZFKOQSIY
https://www.scaler.com/topics/classification-in-r/
https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/
https://medium.com/learning-data/which-machine-learning-algorithm-should-i-use-for-my-analysis-962aeff11102 
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html
	https://www.youtube.com/watch?v=zGdXaRug7LI

Estos recursos han servido de base para poder llevar el análisis más lejos y analizar diversas variables y transformaciones que, pensamos podían tener importancia.

- Diversidad generacional en la empresa: 
  https://onetalent.es/apostando-por-la-diversidad-generacional-en-el-mundo-laboral/
