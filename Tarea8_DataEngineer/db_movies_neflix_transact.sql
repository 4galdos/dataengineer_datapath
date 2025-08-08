CREATE DATABASE db_movies_netflix_transact;
USE db_movies_netflix_transact;


/* CREAMOS LA TABLA Movie */
CREATE TABLE Movie (
  movieID  VARCHAR(8) PRIMARY KEY  NOT NULL,
  movieTitle VARCHAR(100) NOT NULL,
  releaseDate  DATE NOT NULL,
  originalLanguage  VARCHAR(100) DEFAULT NULL,
  link VARCHAR(50) DEFAULT NULL
);

INSERT INTO Movie VALUES ("80192187","Triple Frontier","2019-04-12","English","https://www.netflix.com/pe-en/title/80192187"),
							("81157374","Run","2021-05-21","English","https://www.netflix.com/pe-en/title/81157374"),
                             ("80210920","The Mother","2023-01-05","English","https://www.netflix.com/pe-en/title/80210920");

/* CREAMOS LA TABLA Person */
CREATE TABLE Person (
  personID  VARCHAR(8) PRIMARY KEY  NOT NULL,
  name VARCHAR(100) NOT NULL,
  birthday  DATE NOT NULL
);

INSERT INTO Person VALUES ("72129839","Joseph Chavez Pineda","1997-04-12"),
							("73235434","aria Lopez Gutierrez","1987-05-21"),
                             ("20432364","Maria Alejandra Navarro","1967-01-05");


#CREAMOS LA TABLA Participant
CREATE TABLE Participant (
	movieId VARCHAR(8) PRIMARY KEY NOT NULL,
    personId VARCHAR(8),
    participantRole VARCHAR(30),
    
     CONSTRAINT fk_movie_participant FOREIGN KEY (movieId) REFERENCES Movie (movieID),
     CONSTRAINT fk_movie_person FOREIGN KEY (personId) REFERENCES Person (personId)
);

# insertando valores a la tabla participant
INSERT INTO Participant VALUES 	("80192187","72129839","Actor"),
								("81157374","73235434","Director"),
								("80210920","20432364","Actor");



## CREANDO LA TABLA Gender
CREATE TABLE Gender (
  genderId INTEGER PRIMARY KEY NOT NULL,
  name VARCHAR(100) NOT NULL
);

# insertamos valores a la Gender
INSERT INTO Gender VALUES (1,"Action"),
							(2,"Adventure"),
                            (3,"Drama");
                            

## creamos la tabla moview_gender
CREATE TABLE Movie_Gender (
  movieId varchar(8) PRIMARY KEY NOT NULL,
  genderId INTEGER,
  
  CONSTRAINT fk_Movie_Gender_Movie FOREIGN KEY (movieId) REFERENCES Movie (movieID),
	CONSTRAINT fk_Movie_Gender_Gender FOREIGN KEY (genderId) REFERENCES Gender (genderId)
);

#insertamos valores a la tabla moview_gender
INSERT INTO Movie_Gender VALUES ("80192187",1),
							("81157374",2),
                            ("80210920",3);