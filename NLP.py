import pandas as pd
import numpy as np
import re
import nltk
from flask import json
nltk.download('punkt')
nltk.download('stopwords') # dict of stopwords
nltk.download('averaged_perceptron_tagger') # to enable pos tagging
nltk.download('wordnet') # For WordNetLemmatizer

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

file = open("body_category.json")
file2 = open("processed_symptoms.json")
diseaseCategory = json.load(file)
symptoms = json.load(file2)

# Helper Functions

severity = {
    "high": ["very", "extremely", "quite", "too", "really", "pretty", "severe", "serious"],
    "mild": ["fairly", "slightly", "mild", "slight"]
}
lexeme_dict = {'feverish':'fever'}

category = {
    "Head, face and neck": ["Head", "Face", "Forehead", "Eyes", "Ears", "Nose", "Mouth", "Lips", "Teeth", "Tongue", "Cheeks", "Chin", "Neck"],
    "Skin, hair and nails": ["Skin", "Hair", "Scalp", "Nails", "Cuticles", "Pores", "Epidermis", "Dermis"],
    "Stomach and sides": ["Stomach", "Abdomen", "Belly", "Waist", "Sides", "Obliques"],
    "Upper body": ["Chest", "Shoulders", "Arms", "Biceps", "Triceps", "Forearms", "Elbows", "Wrists", "body"],
    "Lower body": ["Hips", "Glutes", "Thighs", "Hamstrings", "Quadriceps", "Knees", "Shins", "Calves", "Ankles", "Feet", "Toes", "Legs", "body"]
}

# Get the severity of the symptom from user's description
def getSeverity(description):
    split = word_tokenize(description) # Split the words 
    tagged = nltk.pos_tag(split) # Append the parts of speech (pos)
    adv = [ word[0] for word in tagged if word[1] == "RB" or word[1] == "JJ"] # Filter adverbs and adjectives
    impacts = [ level for rb in adv for level in severity if rb in severity[level]] # Determine class of severity
    impact = [impacts if len(impacts) > 0 else "normal"]
    
    return np.unique(impact)

# Function to filter found symptoms by severity
def filterBySeverity(symptoms, severity):
    filtered = [ symptom for symptom in symptoms if severity[0] in symptom] # Filter suggested symptoms by severity
    if len(filtered) > 0:
        return filtered
    else:
        return symptoms
    
# Function to return wordnet label for the parts of speech

# Lemmatization approaches with examples.
# Prakharr0y, GeeksforGeeks, 2022
# For the pos_tag pos parameter, identify the appropriate character.
# https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

def getTag(word):    
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:         
        return None
    
# Function to tokenize the provided sentence
def tokenize(sentence):
    words = word_tokenize(sentence)
    return words

# Function to tag each word with the appropriate part of speech
def tagWords(words):
    tagged = nltk.pos_tag(words)
    return tagged

# Function to remove stop words 
def rmStopwords(words):
    filtered_words = []
    for word in words:
        current = word[0]
        if current.casefold() not in stop_words:
            filtered_words.append((word[0],word[1]))
    return filtered_words

# Function to get the Nouns from the sentence 
def getSubjects(tagged):
    filtered = [word for word in tagged if word[1].startswith('N')]
    return filtered

# Function to break down selected nouns to lexeme
def lemmatizeWords(filtered):
    lemmatized = [lemmatizer.lemmatize(word[0], pos = getTag(word[1])) for word in filtered]
    return lemmatized

# This functions tries to find a match for all words in the lemmatized vars in the symptoms df
def findMatch(lemmatized, severity, all_symptoms):
    matches = []
    possible_symptoms = []
    for word in lemmatized:
        match_count = 0
        for symptom in all_symptoms:
            if re.search(word,symptom):
                if (match_count > 0):
                    possible_symptoms.append(symptom)
                    possible_symptoms = filterBySeverity(possible_symptoms, severity)
                else:
                    possible_symptoms.append(symptom)
                    match_count += 1
        matches.extend(possible_symptoms)
    return matches

# This function serves as the main function to call all others
def process(sentence, all_symptoms):
    severity = getSeverity(sentence)
    split = tokenize(sentence)
    tagged = tagWords(split)
    stops = rmStopwords(tagged)
    subjects = getSubjects(stops)
    lemmatized = lemmatizeWords(subjects)
    matches = findMatch(lemmatized, severity, all_symptoms)
    
    return np.unique(matches)

# This function retrives the symptoms based on the category
def retrieveSymptoms(area):
    list_of_symptoms = []
    if (area != ""):
        list = diseaseCategory[area].split(',')
        for disease in list:
            symptomsList = symptoms[disease].split(',')
            for symptom in symptomsList:
                list_of_symptoms.append(symptom)
    return list_of_symptoms

# filter the symptoms based on keywords in the user's complaint
def symptomList(complaint):
    split_complaint = tokenize(complaint)
    filtered = [word for word in split_complaint if word not in stop_words]

    select_area = ""
    for word in filtered:
        for area in category:
            for part in category[area]:
                symptom_ = lemmatizer.lemmatize(part).casefold()
                
                if re.search(word,symptom_):
                    select_area = area
                    
                    retrieved = retrieveSymptoms(select_area)
                    return retrieved

def formatSymptoms(symptoms):
    formatted = []
    for symptom in symptoms:
        temp_symptom = symptom.replace(" ","_")
        formatted.append(temp_symptom)
    return formatted

def inCategory(nouns):
    found_noun = ""
    for noun in nouns:
        for area in category:
            parts = [ part.lower() for part in category[area] ]
            count = parts.count(noun)
            if count > 0:
                return category[area]

def filterBySymptomDescription(suggested_symptoms, symptom_df, user_description):
    filtered_symptoms = []
    split_complaint = tokenize(user_description)
    filtered_sentence = [word for word in split_complaint if word not in stop_words]
    tagged = tagWords(filtered_sentence)
    Nouns = [ word[0] for word in tagged if word[1] == "NN"]
    nouns_in_category = inCategory(Nouns)

    for noun in nouns_in_category:
        for symptom in suggested_symptoms:
            formatted = symptom.replace("_"," ")
            symptom_name = symptom_df['Symptoms'].tolist()
            symptom_description = symptom_df['Description'].tolist()
            index = symptom_name.index(formatted)
            description = symptom_description[index]

            if re.search(noun.casefold(), description):
                filtered_symptoms.append(symptom)
                
    return filtered_symptoms



# This function handles multiple complaints separated by 'and'
def processLanguage(user_description, symptoms_df):
    all_symptoms = symptoms_df['Symptoms'].tolist()

    sentences = user_description.split('and') # Splits the sentence

    suggested_symptoms = []
    for sentence in sentences: # Loops through each sentence split

        filtered_symptoms_list = symptomList(sentence) if symptomList(sentence) else ""
        select_symptoms_list = []

        if (filtered_symptoms_list == ""):
            select_symptoms_list.extend(all_symptoms)
        else:
            select_symptoms_list.extend(filtered_symptoms_list)
        
        list_of_symptoms = process(sentence.strip(), select_symptoms_list) # Calls the process function for each sentence
        
        try:
            narrowed_down_list = filterBySymptomDescription(list_of_symptoms, symptoms_df, sentence)
            suggested_symptoms.extend(narrowed_down_list)
        except:
            suggested_symptoms.extend(list_of_symptoms)
    
    if len(suggested_symptoms) > 0:
        return formatSymptoms(suggested_symptoms)
    else:
        return ["Inconclusive"]
    
