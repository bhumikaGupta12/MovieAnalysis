from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, regexp_extract, avg
from pyspark.sql.types import IntegerType, StringType, LongType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType , LongType


spark = SparkSession.builder \
    .appName("MovieRatingsAnalysis") \
    .config("spark.sql.shuffle.partitions", "2000") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

ratings_schema = StructType([ 
    StructField('UserID', 
                IntegerType(), True), 
    StructField('MovieID', 
                IntegerType(), True), 
    StructField('Rating', 
                IntegerType(), True), 
    StructField('Timestamp', 
                LongType(), True)
]) 
users_schema = StructType([ 
    StructField('UserID', 
                IntegerType(), nullable=False), 
    StructField('Gender', 
                StringType(), True), 
    StructField('Age', 
                IntegerType(), True), 
    StructField('Occupation', 
                IntegerType(), True),
	StructField('Zip-Code', 
                StringType(), True)
])

movies_schema = StructType([ 
    StructField('MovieID', 
                IntegerType(), nullable=False), 
    StructField('Title', 
                StringType(), True), 
    StructField('Genres', 
                StringType(), True)
])


ratings_rdd= spark.sparkContext.textFile("ratings.dat")
ratings_split = ratings_rdd.map(lambda line : line.split("::")).map(lambda fields: (int(fields[0]), int(fields[1]), int(fields[2]), long(fields[3])))
ratings= spark.createDataFrame(ratings_split, schema= ratings_schema)

users_rdd= spark.sparkContext.textFile("users.dat")
users_split = users_rdd.map(lambda line : line.split("::")).map(lambda fields: (int(fields[0]), fields[1], int(fields[2]), int(fields[3]), fields[4]))
users= spark.createDataFrame(users_split, schema= users_schema)

movies_rdd = spark.sparkContext.textFile("movies.dat")
movies_split = movies_rdd.map(lambda line : line.split("::")).map(lambda fields : (int(fields[0]), fields[1], fields[2]))
movies= spark.createDataFrame(movies_split, schema= movies_schema)

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
