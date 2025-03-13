wikimultihopqa_exps = [
    {
        'question': "When did the director of film Hypocrite (Film) die?",
        'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
        'answer': "19 June 2013",
    },
    {
        'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
        'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
        'answer': "no",
    },
    {
        'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
        'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
        'answer': "no",
    },
    {
        'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
        'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
        'answer': "Genghis Khan",
    },
    {
        'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
        'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
        'answer': "Martin Hodge",
    },
    {
        'question': "When did the director of film Laughter In Hell die?",
        'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
        'answer': "August 25, 1963",
    },
    {
        'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
        'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
        'answer': "Twenty Plus Two",
    },
    {
        'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
        'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
        'answer': "Prithvipati Shah",
    }
]


hotpotqa_exps = [
    {
        'question': "Jeremy Theobald and Christopher Nolan share what profession?",
        'cot': "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
        'answer': "producer",
    },
    {
        'question': "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
        'cot': "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
        'answer': "The Phantom Hour.",
    },
    {
        'question': "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
        'cot': "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
        'answer': "20",
    },
    {
        'question': "Were Lonny and Allure both founded in the 1990s?",
        'cot': "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.", 
        'answer': "no",
    },
    {
        'question': "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
        'cot': "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
        'answer': "Scott Glenn",
    },
    {
        'question': "What was the 2014 population of the city where Lake Wales Medical Center is located?",
        'cot': "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
        'answer': "15,140",
    },
    {
        'question': "Who was born first? Jan de Bont or Raoul Walsh?",
        'cot': "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
        'answer': "Raoul Walsh",
    },
    {
        'question': "In what country was Lost Gravity manufactured?",
        'cot': "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
        'answer': "Germany",
    },
    {
        'question': "Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?",
        'cot': "The debut album of the band \"Hot Hot Heat\" was \"Make Up the Breakdown\". The debut album of the band \"The Operation M.D.\" was \"We Have an Emergency\".",
        'answer': "The Operation M.D.",
    },
    {
        'question': "How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?",
        'cot': "The singer of \"A Girl Like Me\" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award.",
        'answer': "one",
    },
    {
        'question': "The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?",
        'cot': "The actor that stars as Joe Proctor on the series \"Power\" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante.",
        'answer': "Assante",
    },
]


iirc_exps = [
    {
        "question": "What is the age difference between the kicker and the quarterback for the Chargers?",
        "cot": "The kicker for the Chargers is Nate Kaeding. The quarterback (QB) for the Chargers is Philip Rivers. Nate Kaeding was born in the year 1982. Philip Rivers was born in the year 1981. Thus, the age difference between them is of 1 year.",
        "answer": "1"
    },
    {
        "question": "How many years was the ship that took the battalion from New South Wales to Ceylon in service?",
        "cot": "The ship that took the battalion from New South Wales to Ceylon is General Hewitt. General Hewitt was launched in Calcutta in 1811. General Hewitt was sold for a hulk or to be broken up in 1864. So she served for a total of 1864 - 1811 = 53 years.",
        "answer": "53"
    },
    {
        "question": "What year was the theatre that held the 2016 NFL Draft built?",
        "cot": "The theatre that held the 2016 NFL Draft is Auditorium Theatre. The Auditorium Theatre was built in 1889.",
        "answer": "1889"
    },
    {
        "question": "How long had Milan been established by the year that Nava returned there as a reserve in the first team's defense?",
        "cot": "Nava returned to Milan as a reserve in the first team's defense in the year 1990. Milan had been established in the year 1899. Thus, Milan had been established for 1990 - 1899 = 91 years when Milan returned to Milan as a reserve in the first team's defense.",
        "answer": "91"
    },
    {
        "question": "When was the town Scott was born in founded?",
        "cot": "Scott was born in the town of Cooksville, Illinois. Cooksville was founded in the year 1882.",
        "answer": "1882"
    },
    {
        "question": "In what country did Wright leave the French privateers?",
        "cot": "Wright left the French privateers in Bluefield's river. Bluefields is the capital of the South Caribbean Autonomous Region (RAAS) in the country of Nicaragua.",
        "answer": "Nicaragua"
    },
    {
        "question": "Who plays the A-Team character that Dr. Hibbert fashioned his hair after?",
        "cot": "Dr. Hibbert fashioned his hair after Mr. T from The A-Team. Mr T.'s birthname is Lawrence Tureaud.",
        "answer": "Lawrence Tureaud"
    },
    {
        "question": "How many people attended the conference held near Berlin in January 1942?",
        "cot": "The conference held near Berlin in January 1942 is Wannsee Conference. Wannsee Conference was attended by 15 people.",
        "answer": "15"
    },
    {
        "question": "When did the country Ottwalt went into exile in founded?",
        "cot": "Ottwalt went into exile in the country of Denmark. Denmark has been inhabited since around 12,500 BC.",
        "answer": "12,500 BC"
    },
    {
        "question": "When was the J2 club Uki played for in 2001 founded?",
        "cot": "The J2 club that Uki played for is Montedio Yamagata. Montedio Yamagata was founded in 1984.",
        "answer": "1984"
    },
    {
        "question": "When was the person who produced A Little Ain't Enough born?",
        "cot": "A Little Ain't Enough was produced by Bob Rock. Bob Rock was born on April 19, 1954.",
        "answer": "April 19, 1954"
    },
    {
        "question": "Which of the schools Fiser is affiliated with was founded first?",
        "cot": "The schools that Fiser is affiliated with (1) Academy of Music, University of Zagreb (2) Mozarteum University of Salzburg (3) Croatian Music Institute orchestra. Academy of Music, University of Zagreb was founded in the year 1829. Mozarteum University of Salzburg was founded in the year 1841. Croatian Music Institute was founded in the year 1827. Thus, the school founded earliest of these is Croatian Music Institute.",
        "answer": "Croatian Music Institute"
    },
    {
        "question": "How many casualties were there at the battle that Dearing fought at under Jubal Early?",
        "cot": "Under Jubal Early, Dearing fought the First Battle of Bull Run. First Battle of Bull Run has 460 union casualties and 387 confederate casualties. Thus, in total the First Battle of Bull Run had 460 + 387 = 847 casualties.",
        "answer": "847"
    },
    {
        "question": "Which of the two congregations which provided leadership to the Pilgrims was founded first?",
        "cot": "The congregations which provided leadership to the Pilgrims are Brownists and Separatist Puritans. Brownist was founded in 1581. The Separatist Puritans was founded in 1640. Thus, Brownist was founded first.",
        "answer": "Brownist"
    },
    {
        "question": "How long had the Rock and Roll Hall of Fame been open when the band was inducted into it?",
        "cot": "The band was inducted into Rock and Roll Hall of Fame in the year 2017. Rock and Roll Hall of Fame was established in the year of 1983. Thus, Rock and Roll Hall of Fame been open for 2018 - 1983 = 34 years when the band was inducted into it.",
        "answer": "34"
    },
    {
        "question": "Did the Lord Sewer who was appointed at the 1509 coronation live longer than his king?",
        "cot": "Lord Sewer who was appointed at the 1509 coronation was Robert Radcliffe, 1st Earl of Sussex. Lord Sever's king in 1509 was Henry VIII of England. Robert Radcliffe, 1st Earl of Sussex was born in the year 1483, and died in the year 1542. So Robert lived for 1542 - 1483 = 59 years. Henry VIII of England was born in the year 1491 and died in the year 1547. So Henry VIII lived for 1547 - 1491 = 56 years. Thus, Robert Radcliffe lived longer than Henry VIII.",
        "answer": "yes"
    },
    {
        "question": "When was the place near where Manuchar was defeated by Qvarqvare established?",
        "cot": "Manuchar was defeated by Qvarqvare near Erzurum. Erzurum was founded during the Urartian period.",
        "answer": "Urartian period"
    },
    {
        "question": "What year was the man who implemented the 46 calendar reform born?",
        "cot": "The man who implemented the 46 calendar reform is Julius Caesar. Julius Caesar was born in the year 100 BC.",
        "answer": "100 BC"
    },
    {
        "question": "How many years after the first recorded Tommy John surgery did Scott Baker undergo his?",
        "cot": "The first recorded Tommy John surgery happened when it was invented in the year 1974. Scott Baker underwent Tommy John surgery in the year 2012. Thus, Scott Baker underwent Tommy John surgery 2012 - 1974 = 38 years after it was first recorded.",
        "answer": "38"
    },
    {
        "question": "Which was the older of the two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK?",
        "cot": "The two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK are Koudas and Matzourakis. Koudas was born on 23 November 1946. Matzourakis was born on 6 June 1949. Thus, the older person among the two is Koudas.",
        "answer": "Koudas"
    }
]