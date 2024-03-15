# Potato Disease Detection

// Insert contents

## Requirements and Dependencies
Read [requirements.txt](requirements.txt)

## Installation
1. Create a virtual environment using pip and source it
2. Run `pip install requirements.txt`
3. Insert database and table details for user authentication in [api/main.py] under # AUTHENTICATION in variable SQLALCHEMY_DATABASE_URL
4. Insert database details and tablename in [api/main.py](api/main.py) under # DATABASE CONFIG
5. Create the database with the details mentioned above
    - `CREATE DATABASE <dbname>`
    - `USE <dbname>`
    - `CREATE TABLE IF NOT EXISTS table_name (
            ID INT(11) NOT NULL PRIMARY KEY,
            Name VARCHAR(100),
            Causes VARCHAR(1000),
            Symptoms VARCHAR(1000),
            Treatment VARCHAR(1000)
            );`
```sql
INSERT INTO table_name (ID, Name, Causes, Symptoms, Treatment) VALUES (1, 'Early Blight', 'Early blight is caused by the fungus Alternaria solani, which can survive in soil and plant debris, infecting new plants through spores. Warm and humid conditions favor the growth and spread of the fungus. Poor air circulation and high humidity in the canopy of potato plants can contribute to disease development. Overcrowding of plants, excessive nitrogen fertilization, and poor soil drainage can create conditions conducive to early blight.', 'Leaf Lesions; Stem and Tuber Lesions; Leaf Yellowing and Defoliation; Reduced Yield', 'Rotate crops to reduce the buildup of fungal spores in the soil. Use proper spacing and avoid overcrowding of plants to improve air circulation. Avoid overhead irrigation to reduce leaf wetness and humidity. Fungicides containing chlorothalonil, mancozeb, or copper can be effective against early blight. Remove and destroy infected plant debris to reduce the source of fungal spores.');
```

```sql
INSERT INTO table_name (ID, Name, Causes, Symptoms, Treatment) VALUES(2, 'Late Blight', 'Late blight is caused by the oomycete pathogen Phytophthora infestans, which can survive in soil and plant debris, infecting new plants through spores. Cool and wet conditions favor the growth and spread of the pathogen. Rainy weather and high humidity can create conditions conducive to late blight. Overcrowding of plants, excessive nitrogen fertilization, and poor soil drainage can create conditions conducive to late blight.', 'The most characteristic symptom of late blight is the appearance of large, irregularly shaped, water-soaked lesions on the leaves. Lesions can quickly expand and turn brown or black, leading to the collapse of the entire leaf. Lesions can also appear on stems, leading to stem cankers and wilting of the plant. Severe infections can lead to reduced yield and quality of potato tubers.', 'Rotate crops to reduce the buildup of fungal spores in the soil. Use proper spacing and avoid overcrowding of plants to improve air circulation. Fungicides containing chlorothalonil, mancozeb, or copper can be effective against late blight. Remove and destroy infected plant debris to reduce the source of fungal spores.');
```
6. To run backend alone, change directory to api. Run `python3 main.py` and use the endpoint "predict" to predict using a file as input. Can use docs endpoint for graphical interface for sending file.

Note: If using Postman to send file, name the key of the file as "file"

