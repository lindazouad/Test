package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object Preprocessor {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()
    import spark.implicits._


    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/

    /* a. Charger le fichier csv dans un dataFrame. La première ligne du fichier donne le nom de chaque colonne,
    on veut que cette ligne soit utilisée pour nommer les colonnes du dataFrame.
    On veut également que les “false” soient importés comme des valeurs nulles.*/

    val df_input = spark.read.option("header",true).option("nullvalue", "false").csv("/Users/lindazouad/Downloads/train.csv")

    /* b. Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame. */

    val rowNum = df_input.count()
    /*val colNum = df_input.columns.length()*/

    println("Nombre de lignes = " + rowNum)

    /* c. Afficher le dataFrame sous forme de table. */

    df_input.show()

    /* d. Afficher le schéma du dataFrame (nom des colonnes et le type des données contenues dans chacune d’elles). */

    df_input.printSchema()

    /* e. Assigner le type “Int” aux colonnes qui vous semblent contenir des entiers. */

    /*val toInt = udf[Int,String](_.toInt)*/
    val df = df_input
      .withColumn("backers_count", 'backers_count.cast("Int"))
      .withColumn("final_status", 'final_status.cast("Int"))
      .withColumn("goal", 'goal.cast("Int"))
      .select("project_id","name", "desc", "goal", "keywords", "disable_communication", "country", "currency","deadline", "state_changed_at", "created_at", "launched_at", "backers_count", "final_status").toDF()
    df.printSchema()


    /** 2 - CLEANING **/

    /* a. Afficher une description statistique des colonnes de type Int (avec .describe().show )*/

    df.describe("backers_count", "final_status", "goal").show()

    /* b. Observer les autres colonnes, et proposer des cleanings à faire sur les données: faites des groupBy count, des show, des dropDuplicates.
    Quels cleaning faire pour chaque colonne ? Y a-t-il des colonnes inutiles ?
    Comment traiter les valeurs manquantes ? Des “fuites du futur” ??? */

    /* c. enlever la colonne "disable_communication".
    Cette colonne est très largement majoritairement à "false", il y a 311 "true" (négligeable) le reste est non-identifié. */

    val df_c = df.drop("disable_communication")

    /* d. Les fuites du futur: dans les datasets construits a posteriori des évènements, il arrive que des données ne pouvant être connues
    qu'après la résolution de chaque évènement soient insérées dans le dataset. On a des fuites depuis le futur !
    Par exemple, on a ici le nombre de "backers" dans la colonne "backers_count".
    Il s'agit du nombre de personnes FINAL ayant investi dans chaque projet, or ce nombre n'est connu qu'après la fin de la campagne.
    Il faut savoir repérer et traiter ces données pour plusieurs raisons:
      En pratique quand on voudra appliquer notre modèle, les données du futur ne sont pas présentes (puisqu'elles ne sont pas encore connues).
      On ne peut donc pas les utiliser comme input pour un modèle.
    Pendant l'entraînement (si on ne les a pas enlevées) elles facilitent le travail du modèle puisque qu'elles contiennent des informations directement liées à ce qu'on veut prédire.
    Par exemple, si backers_count = 0 on est sûr que la campagne a raté.

Ici, pour enlever les données du futur on retir les colonnes "backers_count" et "state_changed_at".*/

    val df_d = df_c.drop("backers_count", "state_changed_at")

    /* e. Créer deux udfs udf_country et udf_currency telles que:
          - Udf_country : si country=null prendre la valeur de currency, sinon laisser la valeur country actuelle.
          On veut les résultat dans une nouvelle colonne “country2”.
          - Udf_currency: si currency.length != 3 currency prend la valeur null, sinon laisser la valeur currency actuelle.
          On veut les résultats dans une nouvelle colonne “currency2”.*/

    def udf_country = udf{(country: String, currency: String) =>
      if (country == null)
        currency
      else
        country
    }

    def udf_currency = udf{(currency: String) =>
      if (currency != null && currency.length() != 3)
        null
      else
        currency
    }


    val df_e1 = df_d.withColumn("country2", udf_country($"country", $"currency"))
    val df_e2 = df_e1.withColumn("currency2",udf_currency($"currency"))

    /*withColumn("currency2",udf_currency($"currency"))*/


    /* f. Pour une classification, l’équilibrage entre les différentes classes cibles dans les données d’entraînement doit être contrôlé (et éventuellement corrigé).
    Afficher le nombre d’éléments de chaque classe (colonne final_status). */


    val df_f = df_e2.groupBy("final_status").count().orderBy($"count".desc)


    /* g. Conserver uniquement les lignes qui nous intéressent pour le modèle: final_status = 0 (Fail) ou 1 (Success).
    Les autres valeurs ne sont pas définies et on les enlève.
    On pourrait toutefois tester en mettant toutes les autres valeurs à 0 en considérant que les campagnes qui ne sont pas un Success sont un Fail.*/


    val df_clean = df_e2.filter("final_status in (0,1)")


    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/

    /** 3 - Ajouter et manipuler des colonnes **/

    /* a. Ajoutez une colonne days_campaign qui représente la durée de la campagne en jours
    (le nombre de jours entre “launched_at” et “deadline”).*/

    val df_3a  = df_clean.withColumn("days_campaign", $"deadline" - $"launched_at")

    /* b. Ajoutez une colonne hours_prepa qui représente le nombre d’heures de préparation de la campagne entre “created_at” et “launched_at”.
    On pourra arrondir le résultat à 3 chiffres après la virgule.*/

    val df_3b  = df_3a.withColumn("hours_prepa",  $"launched_at" - $"created_at")

    /* c. Supprimer les colonnes “launched_at”, “created_at” et “deadline”, elles ne sont pas exploitables pour un modèle. */

    val df_3c = df_3b.drop("launched_at", "created_at", "deadline")

    /* d. Ajoutez une colonne text, qui contient la concaténation des Strings des colonnes “name”, “desc” et “keywords”.
    ATTENTION à bien mettre des espaces entre les chaînes de caractères concaténées,
    car on fera plus un split en se servant des espaces entre les mots.*/

    val df_3d = df_3c.withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    /** 4 - Valeurs nulles**/

    /* a. Remplacer les valeurs nulles des colonnes "days_campaign”, "hours_prepa", "goal" par la valeur -1.*/


    val df_4a = df_3d.na.fill(Map("name" -> -1, "hours_prepa" -> -1, "goal" -> -1))
    df_4a.show()

    /*Enregistrement des données :*/

    df_4a.write.mode(SaveMode.Overwrite).parquet("/Users/lindazouad/Downloads/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")












  }

}
