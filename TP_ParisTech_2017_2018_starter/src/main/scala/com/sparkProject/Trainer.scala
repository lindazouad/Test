package com.sparkProject
//Chargement des packages nécessaires :

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, SQLImplicits, SQLContext}
import org.apache.spark.ml.{feature, classification, evaluation, tuning, Pipeline}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, StopWordsRemover, CountVectorizer, CountVectorizerModel,
IDF, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.PipelineModel

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

     //1. Charger de la dataframe
   val df = spark.read.parquet("/Users/lindazouad/Downloads/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")

    // TF-IDF

    // 2. Utiliser les données textuelles

    /* a. 1er stage: La première étape est séparer les textes en mots (ou tokens) avec un tokenizer.
    Vous allez donc construire le premier Stage du pipeline en faisant:*/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    /* b. 2e stage: On veut retirer les stop words pour ne pas encombrer le modèle avec des mots qui ne véhiculent pas de sens.
    Créer le 2ème stage avec la classe StopWordsRemover.*/

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words_removed")


    /* c. 3e stage: La partie TF de TF-IDF est faite avec la classe CountVectorizer.*/

    val vectorizer = new CountVectorizer()
      .setInputCol("words_removed")
      .setOutputCol("Vectorized")


    /* d. 4e stage: Trouvez la partie IDF. On veut écrire l’output de cette étape dans une colonne “tfidf”.*/

    val idf = new IDF()
      .setInputCol("Vectorized")
      .setOutputCol("tfidf")

    // 3. Convertir les catégories en données numériques

    /* e. 5e stage: Convertir la variable catégorielle “country2” en données numérique.
    On veut les résultats dans une colonne "country_indexed".*/


    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    /* 6e stage: Convertir la variable catégorielle “currency2” en données numérique.
    On veut les résultats dans une colonne "currency_indexed".*/

    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")


    // 4. Mettre les données sous une forme utilisable par Spark.ML.

    /* g. 7e stage: Assembler les features "tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"
      dans une seule colonne “features”.*/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /*val df_g = assembler.transform(df)*/

    /* h. 8e stage: Le modèle de classification, il s’agit d’une régression logistique que vous définirez de la façon suivante: */

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /* i. Enfin, créer le pipeline en assemblant les 8 stages définis précédemment, dans le bon ordre.*/


    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorizer,idf, indexer_country, indexer_currency, assembler, lr))


    // 5. Entraînement et tuning du modèle

    /* j.Créer un dataFrame nommé “training” et un autre nommé “test”  à partir du dataFrame chargé initialement de façon à
    le séparer en training et test sets dans les proportions 90%, 10% respectivement.*/

    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1))

      /*Pour la régularisation de la régression logistique on veut tester les valeurs de 10e-8 à 10e-2 par pas de 2.0
      en échelle logarithmique (on veut tester les valeurs 10e-8, 10e-6, 10e-4 et 10e-2).
      Pour le paramètre minDF de countVectorizer on veut tester les valeurs de 55 à 95 par pas de 20.
      En chaque point de la grille on veut utiliser 70% des données pour l’entraînement et 30% pour la validation.
    On veut utiliser le f1-score pour comparer les différents modèles en chaque point de la grille (https://en.wikipedia.org/wiki/F1_score).
    Cherchez dans ml.evaluation. */

    /* k. Préparer la grid-search pour satisfaire les conditions explicitées ci-dessus
    puis lancer ​la ​grid-search ​sur ​le ​dataset ​“training” ​préparé ​précédemment. */

    // Définition de la grille de paramètres à tester :

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF, Array[Double](55, 75, 95))
      .build()

    // Choix de la métrique d'évaluation du modèle :

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Lancement de la grid-search sur le dataset préparé :


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /* l. Appliquer le meilleur modèle trouvé avec la grid-search aux données test.
    Mettre les résultats dans le dataFrame ​ df_WithPredictions. Afficher le f1-score du modèle ​sur ​les ​données ​de ​test.*/


    // Extraction du meilleur modèle issu de la grid-search :
    val model = trainValidationSplit.fit(training)

    // Application du meilleur modèle aux données test :
    val df_WithPredictions = model.transform(test)

    //Affichage du f1-score obtenu sur les données test :
    val f1_score = evaluator.evaluate(df_WithPredictions)

    println("\n\n\nf1 score du meilleur modèle sur les données test : " + (f1_score) + "\n\n")

    // m. Afficher ​​ df_WithPredictions.groupBy(​ "final_status"​ , ​​ "predictions"​ ).count.show()

    df_WithPredictions.groupBy("final_status", "predictions").count.show()


  }
}
