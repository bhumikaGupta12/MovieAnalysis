from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, regexp_extract, avg
from pyspark.sql.types import IntegerType, StringType, LongType

spark = SparkSession.builder \
    .appName("MovieAnalysis") \
    .config("spark.sql.shuffle.partitions", "2000") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

ratings_schema = ["UserID", "MovieID", "Rating", "Timestamp"]
users_schema = ["UserID", "Gender", "Age", "Occupation", "Zipcode"]
movies_schema = ["MovieID", "Title", "Genres"]

ratings = spark.read.csv("ratings.dat", sep="::", schema=ratings_schema)
users = spark.read.csv("users.dat", sep="::", schema=users_schema)
movies = spark.read.csv("movies.dat", sep="::", schema=movies_schema)

movies = movies.withColumn("Year", regexp_extract(col("Title"), r"\((\d{4})\)", 1).cast(IntegerType()))
movies = movies.filter(col("Year") > 1989)

users = users.filter(col("Age").between(18, 49))

users.cache()
movies.cache()

ratings_users = ratings.join(users, on="UserID", how="inner")
ratings_movies_users = ratings_users.join(movies, on="MovieID", how="inner")

ratings_movies_users = ratings_movies_users.withColumn("Genre", explode(split(col("Genres"), "\\|")))

ratings_movies_users.cache()

average_ratings = ratings_movies_users.groupBy("Year", "Genre").agg(avg("Rating").alias("AverageRating"))

average_ratings = average_ratings.repartition(2000, "Year", "Genre").orderBy("Year", "Genre")

average_ratings.show()

spark.stop()
