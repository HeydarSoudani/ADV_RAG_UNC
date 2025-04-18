rephrased_exps = [
    {
        'question': "In what school district is Governor John R. Rogers High School, named after John Rankin Rogers, located?",
        'Rephrased': "What school district is Governor John R. Rogers High School in?"
    },
    {
        'question': "Which Australian racing driver won the 44-lap race for the Red Bull Racing team?",
        'Rephrased': "Who is the Australian racing driver that won a 44-lap race for the Red Bull Racing team?"
    },
    {
        'question': "What star of *Parks and Recreation* appeared in November?",
        'Rephrased': "Which actor from *Parks and Recreation* made an appearance in November?"
    },
    {
        'question': "Which genus of flowering plant is found in an environment further south, Crocosmia or Cimicifuga?",
        'Rephrased': "Between Crocosmia and Cimicifuga, which plant genus is typically found further south?"
    },
    {
        'question': "In what year did the man who shot the Chris Stockley, of The Dingoes, die?",
        'Rephrased': "What is the year of death for the man who shot Chris Stockley, of The Dingoes?"
    },
]


wikimultihopqa_query_exps = [
    {
        'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
        'subqa': [("Kurram Garhi is located in which country?", "Pakistan"), ("Trojkrsti is located in which country?", "Republic of Macedonia")],
        'answer': "no"
    },
    {
        'question': "When did the director of film Laughter In Hell die?",
        'subqa': [("Who is the director of the film Laughter In Hell?", "Edward L. Cahn"), ("When did the director of the film Laughter In Hell die?", "August 25, 1963")],
        'answer': "August 25, 1963"
    },
    {
        'question': "What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?",
        'subqa': [("Who is the mother of Grand Duke Alexei Alexandrovich of Russia?", "Maria Alexandrovna"), ("What is the cause of death of mother of Grand Duke Alexei Alexandrovich of Russia?", "tuberculosis")],
        'answer': "tuberculosis"
    },
    {
        'question': "Where did the director of film Maddalena (1954 Film) die?",
        'subqa': [("Who is the director of the film Maddalena?", "Augusto Genina"), ("Where did the director of the film Maddalena die?", "Rome")],
        'answer': "Rome"
    },
    {
        'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
        'subqa': [("What is David Dhawan's nationality?", "India"), ("What is Karl Freund's nationality?", "Germany")],
        'answer': "no"
    },
    {
        'question': "Which album was released more recently, If I Have to Stand Alone or Answering Machine Music?",
        'subqa': [("When was If I Have to Stand Alone released?", "1991"), ("When was Answering Machine Music released?", "1999")],
        'answer': "Answering Machine Music"
    },
    {
        'question': "Who is Boraqchin (Wife Of \u00d6gedei)'s father-in-law?",
        'subqa': [("Who is Boraqchin married?", "\u00d6gedei Khan"), ("Who is Boraqchin's father-in-law?", "Genghis Khan")],
        'answer': "Genghis Khan"
    },
    {
        'question': "Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?",
        'subqa': [("Who is the director of the film When The Mad Aunts Arrive?", "Franz Josef Gottlieb"), ("Who is the director of the film The Miracle Worker (1962 film)?", "Arthur Penn"), ("When did the director of the film When The Mad Aunts Arrive die?", "23 July 2006"), ("When did the director of the film The Miracle Worker (1962 film) die?", "September 28, 2010")],
        'answer': "When The Mad Aunts Arrive"
    },
]