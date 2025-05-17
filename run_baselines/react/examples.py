
examples = {
    "webthink_simple6": [
        {
            "question": 'What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?',
            "steps": [
                {
                    "Thought": 'I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
                    "Action": 'Search[Colorado orogeny]',
                    "Observation": 'The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.'
                },
                {
                    "Thought": 'It does not mention the eastern sector. So I need to look up eastern sector.',
                    "Action": 'Lookup[eastern sector]',
                    "Observation": '(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.',
                },
                {
                    "Thought": 'The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.',
                    "Action": 'Search[High Plains]',
                    "Observation": 'High Plains refers to one of two distinct land regions:',
                },
                {
                    "Thought": 'I need to instead search High Plains (United States).',
                    "Action": 'Search[High Plains (United States)]',
                    "Observation": 'The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).',
                },
                {
                    "Thought": 'High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.',
                    "Action": 'Finish[1,800 to 7,000 ft]',
                }
            ]
        },
        {
            "question": 'Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?',
            "steps": [
                {
                    "Thought": 'The question simplifies to \"The Simpsons\" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.',
                    "Action": 'Search[Milhouse]',
                    "Observation": 'Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.',
                },
                {
                    "Thought": 'The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".',
                    "Action": 'Lookup[named after]',
                    "Observation": '(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.',
                },
                {
                    "Thought": 'Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.',
                    "Action": 'Finish[Richard Nixon]',
                }
            ]
        },
        {
            "question": 'Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?',
            "steps": [
                {
                    "Thought": 'I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.',
                    "Action": 'Search[Adam Clayton Powell]',
                    "Observation": "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].",
                },
                {
                    "Thought": 'To find the documentary, I can search Adam Clayton Powell (film).',
                    "Action": 'Search[Adam Clayton Powell (film)]',
                    "Observation": 'Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.',
                },
                {
                    "Thought": 'Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.',
                    "Action": 'Finish[The Saimaa Gesture]',
                },
            ]
            
        },
        {
            "question": 'What profession does Nicholas Ray and Elia Kazan have in common?',
            "steps": [
                {
                    "Thought": 'I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.',
                    "Action": 'Search[Nicholas Ray]',
                    "Observation": 'Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.',
                },
                {
                    "Thought": 'Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.',
                    "Action": 'Search[Elia Kazan]',
                    "Observation": 'Elia Kazan was an American film and theatre director, producer, screenwriter and actor.',
                },
                {
                    "Thought": 'Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.',
                    "Action": 'Finish[director, screenwriter, actor]',
                },
            ]
        },
        {
            "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            "steps": [
                {
                    "Thought": "I need to search Arthur's Magazine and First for Women, and find which was started first.",
                    "Action": "Search[Arthur's Magazine]",
                    "Observation": "Arthur's Magazine (1844-\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century.",
                },
                {
                    "Thought": "Arthur's Magazine was started in 1844. I need to search First for Women next.",
                    "Action": 'Search[First for Women]',
                    "Observation": "First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.",
                },
                {
                    "Thought": "First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.",
                    "Action": "Finish[Arthur's Magazine]",
                },
            ]
        },
        {
            "question": 'Were Pavel Urysohn and Leonid Levin known for the same type of work?',
            "steps": [
                {
                    "Thought": 'I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.',
                    "Action": 'Search[Pavel Urysohn]',
                    "Observation": 'Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.',
                },
                {
                    "Thought": 'Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.',
                    "Action": 'Search[Leonid Levin]',
                    "Observation": 'Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.',
                },
                {
                    "Thought": 'Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.',
                    "Action": 'Finish[yes]',
                },
            ]
        },
    ]
}
  

# {"webthink_simple": "\n
#  Question: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\n
#  Thought 1: I only need to search Milhouse and find who it is named after.\n
#  Action 1: Search[Milhouse]\n
#  Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening. Milhouse is Bart Simpson's best friend in Mrs. Krabappel's fourth grade class at Springfield Elementary School. He is an insecure, gullible, and less popular child than Bart who is often led into trouble by Bart, who takes advantage of his friend's na\u00c3\u00afvet\u00c3\u00a9. Milhouse is a regular target for school bully Nelson Muntz and his friends Jimbo Jones, Dolph Starbeam and Kearney Zzyzwicz. Milhouse has a crush on Bart's sister, Lisa, a common plot element.\n
#  Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".\n
#  Action 2: Lookup[named after]\n
#  Observation 2: (Result 1 / 1) Milhouse was designed by Matt Groening for a planned series on NBC, which was abandoned.[4] The design was then used for a Butterfinger commercial, and it was decided to use the character in the series.[5][6] Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. The name was the most \"unfortunate name Matt Groening could think of for a kid\".[1] Years earlier, in a 1986 Life in Hell comic entitled \"What to Name the Baby\", Groening listed Milhouse as a name \"no longer recommended\".[7] Milhouse is a favorite among the staff as Al Jean noted \"most of the writers are more like Milhouse than Bart\".[1] His last name was given to him by Howard Gewirtz, a freelance writer who wrote the episode \"Homer Defined\". Gewirtz got the name from one of his wife's friends.[8] According to Arden Myrin and Dana Gould (former writer and co-executive producer of The Simpsons), Rob Cohen (Simpsons writer) inspired Milhouse's look.[2]\nThought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\n
#  Action 3: Finish[Richard Nixon]\n
 
#  Question: Guitars for Wounded Warriors is an album that was recorded in the village in which New York county?\n
#  Thought 1: I need to search Guitars for Wounded Warriors, then find in which New York county it was recorded.\n
#  Action 1: Search[Guitars for Wounded Warriors]\n
#  Observation 1: Guitars for Wounded Warriors is the sixth solo studio album by guitarist Brian Tarquin, released in August 2014 by Cleopatra Records. In 2016, it received a Best Compilation Album nomination by the Independent Music Awards. All tracks were recorded at Tarquin's Jungle Room Studios in New Paltz (village), New York. Being moved by the lack of support for returning veterans through his life Tarquin decided to compose a heavy instrumental rock album as a way to show his appreciation to all veterans. So once again he enlisted top notch players to guest with him on the album, Billy Sheehan on Black Hawk, Ron \"Bumblefoot\" Thal and Reb Beach on Taliban Terror, Steve Morse on Freedom, Chris Poland on Charlie Surfs and Hunting, Chuck Loeb on Escape Kabul, Hal Lindes on Sand & Blood, Gary Hoey on 5 Klicks To Hell and Baghdad, Alex De Rosso Dokken on Broken Arrow, and The Flyin' Ryan Brothers on Surgical Strike.  The entire album was engineered, produced and composed by Tarquin especially for each one of the guest guitarists. Partial proceeds are donated to the Fisher House Foundation from sales.\n
#  Thought 2: Guitars for Wounded Warriors was recorded in New Paltz (village), New York. I need to search New Paltz (village), New York and find which county it is in.\n
#  Action 2: Search[New Paltz (village), New York]\n
#  Observation 2: New Paltz is a village in Ulster County located in the U.S. state of New York. It is approximately 80 miles (130 km) north of New York City and 70 miles (110 km) south of Albany. The population was 7,324 at the 2020 census.[3]\n
#  Thought 3: New Paltz (village), New York is in Ulster County, so the answer is Ulster County.\n
#  Action 3: Finish[Ulster County]\n
 
#  Question: This American actor who starred in \"Veep\" was also in an action comedy film directed by Seth Rogen and Evan Goldberg heavily inspired by what?\n
#  Thought 1: There could be many American actors who starred in \"Veep\". So I can first search \"directed by Seth Rogen and Evan Goldberg\" and find action comedy films directed by Seth Rogen and Evan Goldberg.\n
#  Action 1: Search[directed by Seth Rogen and Evan Goldberg]\n
#  Observation 1: Could not find directed by Seth Rogen and Evan Goldberg. Similar: ['The Interview (redirect from The Interview (Seth Rogen film))', 'Evan Goldberg', 'Jay and Seth Versus the Apocalypse', 'This Is the End (category Films directed by Seth Rogen)', 'Seth Rogen filmography', 'Superbad (category Films with screenplays by Seth Rogen)', 'Pineapple Express (film) (category Films with screenplays by Seth Rogen)', 'Sausage Party (category Films produced by Seth Rogen)', 'Untitled Teenage Mutant Ninja Turtles film (category Films produced by Seth Rogen)', 'The Green Hornet (2011 film) (category Films produced by Seth Rogen)', 'Good Boys (film) (category Films produced by Seth Rogen)', 'Seth Rogen', 'Neighbors 2: Sorority Rising (category Films produced by Seth Rogen)', 'An American Pickle (category Films produced by Seth Rogen)', 'The Watch (2012 film) (category Films with screenplays by Seth Rogen)', 'Neighbors (2014 film) (category Films produced by Seth Rogen)', 'The Night Before (2015 film) (category Films produced by Seth Rogen)', 'Martin Starr', 'Preacher (TV series) (category Television shows directed by Steph Green)', 'Long Shot (2019 film) (category Films produced by Seth Rogen)'].\n
#  Thought 2: I can check search The Interview (Seth Rogen film), Jay and Seth Versus the Apocalypse, This Is the End (category Films directed by Seth Rogen) and find which is an action comedy film directed by Seth Rogen and Evan Goldberg.\n
#  Action 2: Search[The Interview (Seth Rogen film)]\n
#  Observation 2: The Interview is a 2014 action-adventure comedy film co-produced and directed by Seth Rogen and Evan Goldberg in their second directorial work, following This Is the End (2013). The screenplay was written by Dan Sterling, based on a story he co-wrote with Rogen and Goldberg. The film stars Rogen and James Franco as journalists who set up an interview with North Korean leader Kim Jong-un (Randall Park), and are then recruited by the CIA to assassinate him. The film is inspired by a 2012 Vice documentary.\n
#  Thought 3: The Interview is an action comedy film directed by Seth Rogen and Evan Goldberg, and it is inspired by a 2012 Vice documentary. So I can probably stop now without checking other films, and the answer is probably a 2012 Vice documentary.\n
#  Action 3: Finish[a 2012 Vice documentary]\n
 
#  Question: The head coach during the 1997\u201398 Indiana Pacers season retired as a player from what NBA team?\n
#  Thought 1: I can search Indiana Pacers, find the head coach during the 1997\u201398 season, then find from what NBA team it is retired as a player.\n
#  Action 1: Search[Indiana Pacers]\n
#  Observation 1: The Indiana Pacers are an American professional basketball team based in Indianapolis. The Pacers compete in the National Basketball Association (NBA) as a member of the league's Eastern Conference Central Division. The Pacers were first established in 1967 as a member of the American Basketball Association (ABA) and became a member of the NBA in 1976 as a result of the ABA\u00e2\u0080\u0093NBA merger. They play their home games at Gainbridge Fieldhouse. The team is named after the state of Indiana's history with the Indianapolis 500's pace cars and with the harness racing industry.[8][9]\nThought 2: I can look up \"1997\u201398\" to find the head coach during the 1997\u201398 season.\n
#  Action 2: Lookup[1997\u201398]\n
#  Observation 2: No more results.\n
#  Thought 3: I can look up \"1997\" instead to find the head coach during the 1997\u201398 season\n
#  Action 3: Lookup[1997]\n
#  Observation 3: (Result 1 / 5) In the 1997\u00e2\u0080\u009398 NBA season, Indiana native and former Boston Celtics great Larry Bird was hired as head coach. He led the Pacers to a 19-game improvement over the previous season, finishing 58\u00e2\u0080\u009324 \u00e2\u0080\u0093 at the time, the most the franchise had ever won as an NBA team, and tying the 1970\u00e2\u0080\u009371 ABA Pacers for the franchise record. Chris Mullin joined the team in the off-season and immediately became a valuable part of the Pacers lineup\u00e2\u0080\u0094and starting small forward. Assistant coaches Rick Carlisle, in charge of the offense, and Dick Harter, who coached the defense, were key in getting the most out of the Pacers' role players such as Dale Davis, Derrick McKey and a young Antonio Davis. Miller and Rik Smits both made the All-Star team that year, and in the playoffs, the Pacers breezed past the Cleveland Cavaliers and New York Knicks before falling to the Chicago Bulls in a seven-game Eastern Conference Finals.\n
#  Thought 4: In the 1997-98 season, former Boston Celtics great Larry Bird was hired as head coach. So the answer is Boston Celtics.\n
#  Action 4: Finish[Boston Celtics]\n
 
#  Question: What profession does Nicholas Ray and Elia Kazan have in common?\n
#  Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\n
#  Action 1: Search[Nicholas Ray]\n
#  Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause. He is appreciated for many narrative features produced between 1947 and 1963 including They Live By Night, In A Lonely Place, Johnny Guitar, and Bigger Than Life, as well as an experimental work produced throughout the 1970s titled We Can't Go Home Again, which was unfinished at the time of Ray's death.\n
#  Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\n
#  Action 2: Search[Elia Kazan]\n
#  Observation 2: Elia Kazan (/\u02c8i\u02d0li\u0259 k\u0259\u02c8z\u00e6n/;[1][2] born Elias Kazantzoglou (Greek: \u0397\u03bb\u03af\u03b1\u03c2 \u039a\u03b1\u03b6\u03b1\u03bd\u03c4\u03b6\u03cc\u03b3\u03bb\u03bf\u03c5);[3] September 7, 1909 \u2013 September 28, 2003) was an American film and theatre director, producer, screenwriter and actor, described by The New York Times as \"one of the most honored and influential directors in Broadway and Hollywood history\".[4]\nThought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAction 3: Finish[director, screenwriter, actor]\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\nThought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.\nAction 1: Search[Arthur's Magazine]\nObservation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]\nThought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.\nAction 2: Search[First for Women]\nObservation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.[2] In 2011 the circulation of the magazine was 1,310,696 copies.[3]\nThought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\nAction 3: Finish[Arthur's Magazine]\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (\u00d0\u009f\u00d0\u00b0\u00cc\u0081\u00d0\u00b2\u00d0\u00b5\u00d0\u00bb \u00d0\u00a1\u00d0\u00b0\u00d0\u00bc\u00d1\u0083\u00d0\u00b8\u00cc\u0081\u00d0\u00bb\u00d0\u00be\u00d0\u00b2\u00d0\u00b8\u00d1\u0087 \u00d0\u00a3\u00d1\u0080\u00d1\u008b\u00d1\u0081\u00d0\u00be\u00cc\u0081\u00d0\u00bd) (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory, and for developing Urysohn's metrization theorem and Urysohn's lemma, both of which are fundamental results in topology. His name is also commemorated in the terms Urysohn universal space, Fr\u00c3\u00a9chet\u00e2\u0080\u0093Urysohn space, Menger\u00e2\u0080\u0093Urysohn dimension and Urysohn integral equation. He and Pavel Alexandrov formulated the modern definition of compactness in 1923.\nThought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin (/le\u00c9\u00aa.o\u00ca\u008a\u00cb\u0088ni\u00cb\u0090d \u00cb\u0088l\u00c9\u009bv\u00c9\u00aan/ lay-oh-NEED LEV-in; Russian: \u00d0\u009b\u00d0\u00b5\u00d0\u00be\u00d0\u00bd\u00d0\u00b8\u00cc\u0081\u00d0\u00b4 \u00d0\u0090\u00d0\u00bd\u00d0\u00b0\u00d1\u0082\u00d0\u00be\u00cc\u0081\u00d0\u00bb\u00d1\u008c\u00d0\u00b5\u00d0\u00b2\u00d0\u00b8\u00d1\u0087 \u00d0\u009b\u00d0\u00b5\u00cc\u0081\u00d0\u00b2\u00d0\u00b8\u00d0\u00bd; Ukrainian: \u00d0\u009b\u00d0\u00b5\u00d0\u00be\u00d0\u00bd\u00d1\u0096\u00cc\u0081\u00d0\u00b4 \u00d0\u0090\u00d0\u00bd\u00d0\u00b0\u00d1\u0082\u00d0\u00be\u00cc\u0081\u00d0\u00bb\u00d1\u0096\u00d0\u00b9\u00d0\u00be\u00d0\u00b2\u00d0\u00b8\u00d1\u0087 \u00d0\u009b\u00d0\u00b5\u00cc\u0081\u00d0\u00b2\u00d1\u0096\u00d0\u00bd; born November 2, 1948) is a Soviet-American mathematician and computer scientist. \nThought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \nAction 3: Finish[yes]\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nThought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\nAction 1: Search[Adam Clayton Powell]\nObservation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell Jr.', 'Adam Clayton Powell Sr.', 'Adam Clayton Powell IV', 'Adam Clayton Powell III', 'Seventh Avenue (Manhattan) (redirect from Adam Clayton Powell Jr. Boulevard)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito', 'Harlem', 'St. Nicholas Historic District', 'Abyssinian Baptist Church', 'Powell (surname)', 'Adam (musical)', 'Statue of Adam Clayton Powell Jr.', 'List of people from Harlem', 'Adam (given name)', 'Five-Percent Nation', 'List of American liberals'].\nThought 2: I can search Adam Clayton Powell (film) to find the documentary.\nAction 2: Search[Adam Clayton Powell (film)]\nObservation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\n
#  Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\n
#  Action 3: Finish[The Saimaa Gesture]\n",
 
 
#  "cotqa_simple": "\n
 
#  Question: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\n
#  Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\n
#  Answer: Richard Nixon\n
 
#  Question: Guitars for Wounded Warriors is an album that was recorded in the village in which New York county?\n
#  Thought: Let's think step by step. Guitars for Wounded Warriors was recorded at Tarquin's Jungle Room Studios in New Paltz (village), New York. New Paltz is a village in Ulster County located in the U.S. state of New York. So the answer is Ulster County.\n
#  Answer: Ulster County\n
 
#  Question: This American actor who starred in \"Veep\" was also in an action comedy film directed by Seth Rogen and Evan Goldberg heavily inspired by what?\n
#  Thought: Let's think step by step. The Interview is an action comedy film directed by Seth Rogen and Evan Goldberg. It is inspired by a 2012 Vice documentary. So the answer is a 2012 Vice documentary.\n
#  Answer: a 2012 Vice documentary\n
 
#  Question: The head coach during the 1997\u201398 Indiana Pacers season retired as a player from what NBA team?\n
#  Thought: Let's think step by step. The head coach during the 1997\u201398 Indiana Pacers season was Larry Bird. Larry Bird retired as a player from Boston Celtics. So the answer is Boston Celtics.\n
#  Answer: Boston Celtics\n
 
#  Question: What profession does Nicholas Ray and Elia Kazan have in common?\n
#  Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\n
#  Answer: director, screenwriter, actor\n
 
#  Question: Which band was founded first, Hole, the rock band that Courtney Love was a frontwoman of, or The Wolfhounds?\n
#  Thought: Let's think step by step. Hole was founded in 1989. The Wolfhounds was founded in 1985. 1989 (Hole) > 1985 (The Wolfhounds), so The Wolfhounds was founded first.\n
#  Answer: The Wolfhounds\n
 
#  Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n
#  Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.\n
#  Answer: Yes\n
 
#  Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n
#  Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\n
#  Answer: The Saimaa Gesture\n",
 
 
 
#  "webqa_simple": "\n
 
#  Question: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\n
#  Answer: Richard Nixon\n
 
#  Question: Guitars for Wounded Warriors is an album that was recorded in the village in which New York county?\n
#  Answer: Ulster County\n
 
#  Question: This American actor who starred in \"Veep\" was also in an action comedy film directed by Seth Rogen and Evan Goldberg heavily inspired by what?\n
#  Answer: 2012 Vice documentary\n
 
#  Question: The head coach during the 1997\u201398 Indiana Pacers season retired as a player from what NBA team?\n
#  Answer: Boston Celtics\n
 
#  Question: What profession does Nicholas Ray and Elia Kazan have in common?\n
#  Answer: director, screenwriter, actor\n
 
#  Question: Which magazine was started first Arthur's Magazine or First for Women?\n
#  Answer: Arthur's Magazine\n
 
#  Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n
#  Answer: yes\n
 
#  Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n
#  Answer: The Saimaa Gesture\n", 
 
#  "webthink_simple_3": "\n
 
#  Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\n
#  Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\n
#  Action 1: Search[Colorado orogeny]\n
#  Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.\n
#  Thought 2: It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.\n
#  Action 2: Lookup[eastern sector]\n
#  Observation 2: (Result 1 / 1) The Colorado orogen, formerly called the Colorado province, is a >500-km-wide belt of oceanic arc rock (1.78\u20131.65 Ga) that extends southward into New Mexico and composes a major part of the Proterozoic provinces of southwestern United States. This transcontinental collisional event occurred during the  Paleoproterozoic (Statherian Period).[1] The Wyoming sector of the Colorado orogeny was formerly called the Medicine Bow orogeny. The eastern sector extends into the High Plains and is called the Central Plains orogeny. The boundary between the Colorado orogeny and the Wyoming craton is the Cheyenne belt, a 5-km-wide mylonitic shear zone that verges northward. The Cheyenne belt transects and cuts off the south edge of the older Trans-Hudson orogeny.[2]\n
#  Thought 3: The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range.\n
#  Action 3: Search[High Plains]\n
#  Observation 3: High Plains refers to one of two distinct land regions:\n
#  Thought 4: I need to search High Plains (United States).\n
#  Action 4: Search[High Plains (United States)]\n
#  Observation 4: The High Plains are a subregion of the Great Plains, mainly in the Western United States, but also partly in the Midwest states of Nebraska, Kansas, and South Dakota, generally encompassing the western part of the Great Plains before the region reaches the Rocky Mountains. The High Plains are located in eastern Montana, southeastern Wyoming, southwestern South Dakota, western Nebraska, eastern Colorado, western Kansas, eastern New Mexico, western Oklahoma, and to just south of the Texas Panhandle.[4] The southern region of the Western High Plains ecology region contains the geological formation known as Llano Estacado which can be seen from a short distance or on satellite maps.[5] From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\n
#  Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\n
#  Action 5: Finish[1,800 to 7,000 ft]\n
 
#  Question: Guitars for Wounded Warriors is an album that was recorded in the village in which New York county?\n
#  Thought 1: I need to search Guitars for Wounded Warriors, then find in which New York county it was recorded.\n
#  Action 1: Search[Guitars for Wounded Warriors]\n
#  Observation 1: Guitars for Wounded Warriors is the sixth solo studio album by guitarist Brian Tarquin, released in August 2014 by Cleopatra Records. In 2016, it received a Best Compilation Album nomination by the Independent Music Awards. All tracks were recorded at Tarquin's Jungle Room Studios in New Paltz (village), New York. Being moved by the lack of support for returning veterans through his life Tarquin decided to compose a heavy instrumental rock album as a way to show his appreciation to all veterans. So once again he enlisted top notch players to guest with him on the album, Billy Sheehan on Black Hawk, Ron \"Bumblefoot\" Thal and Reb Beach on Taliban Terror, Steve Morse on Freedom, Chris Poland on Charlie Surfs and Hunting, Chuck Loeb on Escape Kabul, Hal Lindes on Sand & Blood, Gary Hoey on 5 Klicks To Hell and Baghdad, Alex De Rosso Dokken on Broken Arrow, and The Flyin' Ryan Brothers on Surgical Strike.  The entire album was engineered, produced and composed by Tarquin especially for each one of the guest guitarists. Partial proceeds are donated to the Fisher House Foundation from sales.\n
#  Thought 2: Guitars for Wounded Warriors was recorded in New Paltz (village), New York. I need to search New Paltz (village), New York and find which county it is in.\n
#  Action 2: Search[New Paltz (village), New York]\n
#  Observation 2: New Paltz is a village in Ulster County located in the U.S. state of New York. It is approximately 80 miles (130 km) north of New York City and 70 miles (110 km) south of Albany. The population was 7,324 at the 2020 census.[3]\n
#  Thought 3: New Paltz (village), New York is in Ulster County, so the answer is Ulster County.\n
#  Action 3: Finish[Ulster County]\n
 
#  Question: Which magazine was started first Arthur's Magazine or First for Women?\n
#  Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.\n
#  Action 1: Search[Arthur's Magazine]\n
#  Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]\n
#  Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.\n
#  Action 2: Search[First for Women]\n
#  Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.[2] In 2011 the circulation of the magazine was 1,310,696 copies.[3]\n
#  Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\n
#  Action 3: Finish[Arthur's Magazine]\n", 
 
 
#  \n", "webact_simple6": "
 
#  Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\n
#  Action 1: Search[Colorado orogeny]\n
#  Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\n
#  Action 2: Lookup[eastern sector]\n
#  Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\n
#  Action 3: Search[High Plains]\n
#  Observation 3: High Plains refers to one of two distinct land regions:\n
#  Action 4: Search[High Plains (United States)]\n
#  Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\n
#  Action 5: Finish[1,800 to 7,000 ft]\n
 
#  Question: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\n
#  Action 1: Search[Milhouse]\n
#  Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\n
#  Action 2: Lookup[named after]\n
#  Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. \n
#  Action 3: Finish[Richard Nixon]\n
 
#  Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n
#  Action 1: Search[Adam Clayton Powell]\n
#  Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].\n
#  Action 2: Search[Adam Clayton Powell (film)]\n
#  Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\n
#  Action 3: Finish[The Saimaa Gesture]\n
 
#  Question: What profession does Nicholas Ray and Elia Kazan have in common?\n
#  Action 1: Search[Nicholas Ray]\n
#  Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\n
#  Action 2: Search[Elia Kazan]\n
#  Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\n
#  Action 3: Finish[director, screenwriter, actor]\n
 
#  Question: Which magazine was started first Arthur's Magazine or First for Women?\n
#  Action 1: Search[Arthur's Magazine]\n
#  Observation 1: Arthur's Magazine (1844-\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. \n
#  Action 2: Search[First for Women]\n
#  Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. \n
#  Action 3: Finish[Arthur's Magazine]\n
 
#  Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n
#  Action 1: Search[Pavel Urysohn]\n
#  Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\n
#  Action 2: Search[Leonid Levin]\n
#  Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \n
#  Action 3: Finish[yes]
 
#  \n\n", "cotqa_simple6": "
 
#  Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\n
#  Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\nAnswer: 1,800 to 7,000 ft\nQuestion: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\nThought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\nAnswer: Richard Nixon\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nThought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\nAnswer: The Saimaa Gesture\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\nThought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAnswer: director, screenwriter, actor\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\nThought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\nAnswer: Arthur's Magazine\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.\nAnswer: Yes\n", "webqa_simple6": "Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\nAnswer: 1,800 to 7,000 ft\nQuestion: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\nAnswer: Richard Nixon\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nAnswer: The Saimaa Gesture\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\nAnswer: director, screenwriter, actor\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\nAnswer: Arthur's Magazine\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nAnswer: Yes\n"}
