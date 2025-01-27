from sentence_transformers import SentenceTransformer
import supabase

# Load a pre-trained SentenceTransformer model
# Generates 768-dimensional embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Supabase client
SUPABASE_URL = "https://djxskdtpvrvprdncsbvt.supabase.co"  # Your Supabase URL
SUPABASE_KEY = ""  # Your Supabase service role key
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


def get_1536_embeddings(sentences):
    """Generate 1536-dimensional embeddings for a list of sentences."""
    sentences_temp = [sentence['content'] for sentence in sentences]
    embeddings_768 = model.encode(sentences_temp).tolist(
    )  # Generate 768-dimensional embeddings
    # Duplicate to reach 1536 dimensions
    embeddings_1536 = [embedding + embedding for embedding in embeddings_768]
    return embeddings_1536


def bulk_push_to_supabase(sentences, embeddings):
    """Insert a batch of content and embeddings into Supabase."""
    rows = [
        {
            "id": sentence["id"],
            "content": sentence["content"],
            "metadata": sentence.get("metadata", {}),
            "embedding": embedding
        }
        for sentence, embedding in zip(sentences, embeddings)
    ]

    # Push rows into Supabase
    response = supabase_client.table("documents").insert(rows).execute()
    return response


sentences = [
    {
        "id": 1,
        "content": "Main Faqs\n\nWhat programming languages and frameworks does the platform cover?\nOur Frontend Developer Career Path uses HTML, CSS and JavaScript before moving on to React.\nWe also have an AI Engineering Path aimed at devs who want to learn about implementing AI solutions. \n\nDoes the platform offer project-based learning where I can build real-world projects?\nYes. All of our projects use real-world examples and all of our learning is project based.",
        "metadata": {}
    },
    {
        "id": 2,
        "content": "What is the format of the courses?\nOur courses use interactive scrims. A scrim is both a video player and a code editor. When you pause the video, you can edit the code the teach was working on. It\u2019s really cool and you have to try it to believe it.\n\nDoes the platform offer live classes or is it self-paced?\nAll of our courses are streamed on demand, so it\u2019s self-paced. We do have a weekly Townhall meet up on Discord and Bootcamp students have online meetings at fixed times.",
        "metadata": {}
    },
    {
        "id": 3,
        "content": "What kind of support is available (forums, mentorship, etc.)?\nIs there a community of learners where I can collaborate and network?\nWe love our thriving Discord community! It\u2019s a great place to get help and meet like minded coders! And we moderate it to make sure it\u2019s a friendly and welcoming community for all.\n\n\nDoes the platform offer certificates or credentials upon completion of a course or program?\nYes, when you graduate you get a certificate.",
        "metadata": {}
    },
    {
        "id": 4,
        "content": "What is the cost of the courses or subscription, and are there any discounts or scholarships available?\nWe use purchasing power parity, so please check Scrimba.com for your local price. If you are struggling to pay, please email us at help@scrimba.com and we will do our best to help.\n\nCan I access the platform and its resources offline?\nAt the moment, we are an online only platform.",
        "metadata": {}
    },
    {
        "id": 5,
        "content": "What are the technical requirements to use the platform (system requirements, internet speed, etc.)?\nScrimba is ultra lightweight so you will probably find you can use it on a low spec PC with mobile internet!",
        "metadata": {}
    },
    {
        "id": 6,
        "content": "Does the platform offer career assistance or job placement services?\nWe have sections of our community and our Frontend Developer Career Path  dedicated to getting you your first job. We advise you on use of repos and portfolio projects, CVs, Resumes, covering letters and LinkedIn. We also have live streams where we interview recruiters.\n\nHow frequently is the content updated to keep up with the latest industry trends and technologies?\nWe are constantly updating and improving our content!",
        "metadata": {}
    },
    {
        "id": 7,
        "content": "Trial Period\n\nDoes the platform offer a trial period or a money-back guarantee to test out the platform before committing? You can try the first scrims of a course for free. After that, we bill monthly but there is no minimum term so you can cancel when you want.\n\nScrimba Bootcamp - Onboarding Resources\n\nWe highly recommend having an intro call with a teacher assistant to onboard you onto the Bootcamp and answer any questions to help you get the most out of the program.",
        "metadata": {}
    },
    {
        "id": 8,
        "content": "Following are helpful resources to help set you up for success as a Scrimba Bootcamp student. If there are any resources that you\u2019d like added to this doc, get in touch on Discord. We look forward to learning with you!\n\nBootcamp Curriculum\nThe Bootcamp follows the curriculum of our Frontend Developer Career Path. Access all the modules and projects of the path on Scrimba.com.",
        "metadata": {}
    },
    {
        "id": 9,
        "content": "Study Groups\nYour Discord Study Group consists of peers who share your goals. Think of it as a friendly space to get help, companionship, and encouragement.",
        "metadata": {}
    },
    {
        "id": 10,
        "content": "When you first join the group, we encourage you to reach out to your group and let them know about you, your background, and your coding journey. It\u2019s important to be active, post daily, and be sure to join our weekly sessions. The more you get involved, the more you\u2019ll benefit from it and get the most out of your Bootcamp learning experience.",
        "metadata": {}
    },
    {
        "id": 11,
        "content": "Common study group posts to help you stay on track: \n\nWeekly learning commitments (\u201dThis week I will...\u201d)\nDaily check-ins (\u201dToday I will... \u201d)\nShare takeaways and highlights about what you\u2019re learning &amp; building\nLearning challenges and wins\nKey takeaways or highlights from your code reviews",
        "metadata": {}
    },
    {
        "id": 12,
        "content": "Virtual Meetups\nYou'll join weekly calls led by a Bootcamp mentor. The purpose of these sessions is to help give your week structure and increase your connection with peers. During these check-ins you can meet peers and your group leader, reflect on the previous week, ask questions, etc.",
        "metadata": {}
    },
    {
        "id": 13,
        "content": "We frequently invite Scrimba teachers, code reviewers, and professional developers to chat with the group and answer questions. You'll also share wins, challenges, or anything you need support with. Each session is usually 1hr long, and you'll choose the session that best fits your schedule. Or feel free to attend all three.",
        "metadata": {}
    },
    {
        "id": 14,
        "content": "Mondays - 4pm CET / 10am EST / 7am PST\nMondays - 8pm CET / 2pm EST / 11am PST\nSaturdays - 5pm CEST / 11am EST / 8am PST\nYou can view all weekly Bootcamp events on your Scrimba dashboard (available at https://scrimba.com/dashboard#overview,), both in the \"Bootcamp\" widget and under \u201cUpcoming events\u201d. You can join the Zoom meetings directly from there by clicking the hyperlinks in the Bootcamp section. ALl times are in times in your local timezone.",
        "metadata": {}
    },
    {
        "id": 15,
        "content": "Coding Workshops\nEach week, Bootcamp mentors are available in the #student-sessions channel at various 1-hour time blocks to better support you. These sessions are tailored towards a particular coding topic or be a more general coding help session. Either way, be sure to take advantage of that hour and drop in if you have questions about what you're learning, are stuck on a project, or want to chat about front-end development.",
        "metadata": {}
    },
    {
        "id": 16,
        "content": "Your #student-sessions Discord channel is also a voice channel where you can have a chat with Bootcamp members at any time. We encourage you to jump into the channel when it's active and share your daily learning goals, what you're working on, any challenges with a course or project, what you're learning, or just catch up with other students and say hi.",
        "metadata": {}
    },
    {
        "id": 17,
        "content": "Code Reviews\nWhen you complete a Solo Project, you will be able to submit it to Scrimba so that one of our vetted teacher's assistants can give you a scrim-based code review. To submit a Solo Project for code review, type /review on any Discord channel and follow the Pumpkin bot's link. \nThe turnaround time is roughly 24 hours after submitting your Solo Project for review. We'll notify you when your code review is complete in the #code-reviews channel on Discord.",
        "metadata": {}
    },
    {
        "id": 18,
        "content": "You must submit your Solo Project as a scrim to get a code review.",
        "metadata": {}
    },
    {
        "id": 19,
        "content": "Follow-up Reviews\n\nYou may ask for a follow-up review to go over some of the changes made based on code reviewer feedback and suggestions; these will likely be shorter and focused more on specific changes.\nFollow-up requests are usually made in the Discord thread for the code review. You don\u2019t necessarily need to submit another form for the review.\n\nGetting Help with Coding Questions",
        "metadata": {}
    },
    {
        "id": 20,
        "content": "You can use the Bootcamp forum #code-help to post questions to the group at large. To best understand and answer your question, you should explain the problem with as much detail as possible, what you\u2019ve tried so far, and - if possible - share a link to your code or the lesson that you have questions about.\nYou may also schedule a 1-1 call with a mentor or teacher to discuss coding questions and walk through it with screen sharing. To request, just submit this form.\n\nResources",
        "metadata": {}
    },
    {
        "id": 21,
        "content": "Resources\n\nBootcamp FAQs\nWhat is a Solo Project? \n\nA Solo Project is one of the projects you'll need to complete as part of the Frontend Developer Career Path. Currently, there are 13+ Solo Projects within the 15 modules, and we're working on adding more soon.\n\nWhere can I find the Bootcamp-exclusive Solo Projects?\nThose Solo Projects are all listed in your #study-group channel\u2019s Pinned Messages.",
        "metadata": {}
    },
    {
        "id": 22,
        "content": "Can I attend all of the Sunday and Monday sessions?\nYou sure can. We encourage you to do so from time-to-time to get to know the group.\n\nIs attendance in the Zoom and/or Discord sessions required?\nNo! These are optional, and we understand if your schedule doesn\u2019t allow you to attend. But these are a great opportunity to connect with others, ask questions, and stay accountable.",
        "metadata": {}
    },
    {
        "id": 23,
        "content": "Are the Bootcamp Monday sessions recorded?\nNot the entire session. We do record the guest intro and Q and A portion of the session when we have one.\n\nRecordings of Guest Q and As from Previous Sessions",
        "metadata": {}
    },
    {
        "id": 24,
        "content": "8/8/22 - Myles Young: Front-end Engineer, Shopify\n8/15/22 - Cameron Blackwood: thecodercareer.com\n8/22/22 - Nadia Zhuk, engineer at Intercom and a volunteer at Women Who Code London and CodeYourFuture\n9/6/22 - Per from Scrimba (CEO)\n9/19/22 - Giovanna Moeller: Software Developer, Content Creator and Bootcamp code reviewer\n10/3/22 - Bob Ziroll: Scrimba Teacher\n10/17/22 - Sara: Scrimba Bootcamp Code Reviewer\n10/25/22 - Per from Scrimba (CEO)\n11/14/22 - Tom Chant: Scrimba Teacher",
        "metadata": {}
    },
    {
        "id": 25,
        "content": "11/14/22 - Tom Chant: Scrimba Teacher\n05/15/23 - Tom Chant from Scrimba\n06/05/23 - Hussien Khayoon: Staff Developer, Shopify\n06/05/23 - Bob Ziroll: Scrimba Instructor, React Superstar\n06/12/23 - Silvia Piovesan, Frontend Developer\n06/19/23 - Gresham Tembo, Frontend Developer\n06/19/23 - Bob Ziroll, Advanced React course preview\n06/26/23 - Rafid Hoda, Scrimba Instructor\n07/03/23 - James Tsetsekas, Fullstack Developer\n07/31/23 - Per Borgen: Scrimba CEO",
        "metadata": {}
    },
    {
        "id": 26,
        "content": "The Frontend Developer Career Path\n\nThis career path will turn you into a hireable frontend developer, and teach you how to nail the job interview. It contains over 70 hours of top-notch tutorials, hundreds of coding challenges, and dozens of real-world projects.\n\nYour program\nThe program contains 12 modules. All modules are filled with interactive coding challenges to ensure that you don't fall off the wagon. You'll learn HTML, CSS, JavaScript, React, UI design, career strategy, and more.",
        "metadata": {}
    },
    {
        "id": 27,
        "content": "Module 1 - 10 lessons - 25 min\nGet prepared. In this module, you'll meet your teachers, learn how Scrimba works, and build your first web app.\n\nModule 2 \u2013 99 lessons - 5 hours 14 min\nWeb dev basics\nLearn the very basics of HTML and CSS. Start creating layouts, and style them how you want.\n\nProjects: Build a Google.com clone, Build a digital Business Card, Build a Space Exploration site, Build a Birthday GIFt Site\n\nSolo Project: Hometown Homepage",
        "metadata": {}
    },
    {
        "id": 28,
        "content": "Solo Project: Hometown Homepage\n\nModule 3 \u2013 209 lessons - 10 hours 21 min\nMaking websites interactive\nCombine your newly acquired HTML & CSS skills with Javascript. This will allow you to create interactive websites.\n\nProjects: Build a passenger counter app, JavaScript challenges - part 1, Set Up a Local Dev Environment, Build a Blackjack game, JavaScript challenges - part 2, Build a Chrome Extension, JavaScript challenges - part 3, Build a Mobile App",
        "metadata": {}
    },
    {
        "id": 29,
        "content": "Module 4 \u2013 64 lessons - 3 hours 56 min\nEssential CSS concepts\nIn this module, you'll level up your CSS skills, and build a neat NFT site.\nProjects: Build an NFT Site, CSS Fundamentals: Challenges, Build a Coworking Space Site\n\nModule 5 \u2013 150 lessons - 9 hours 58 min\nEssential JavaScript concepts\nIn this module, you'll level up your JavaScript skills and build three super-cool apps.",
        "metadata": {}
    },
    {
        "id": 30,
        "content": "Projects: The World's Most Annoying Cookie Consent, Pumpkin's Purrfect Meme Picker, Twimba: Twitter Clone, Essential JS Mini Projects\n\n\n\nModule 6 \u2013 78 lessons - 4 hours 52 min\nResponsive design\nThis module teaches you how to make your websites work well on all screen sizes, a critical skill for any frontend developer.\n\nProjects: Build a Responsive Site, Build a Product Splash Page, CSS Grid: The ultimate layout tool\nSolo Project: Learning Journal",
        "metadata": {}
    },
    {
        "id": 31,
        "content": "Module 7 \u2013 15 lessons - 47 min\nCode reviews\nLearn what code reviews are, why they matter, and how to give successful code reviews.\n\nProjects: Branching and Pull Requests with GitHub Desktop\n\nModule 8 \u2013 95 lessons - 7 hours 17 min\nWorking with APIs\nWeb APIs are the backbone of the web. In this module, you'll learn to use it, and build several different projects.\n\nProjects: Intro to APIs & BoredBot, URLs, REST, & BlogSpace, Async JavaScript & War, Promise Rejection & Capstone",
        "metadata": {}
    },
    {
        "id": 32,
        "content": "Module 9 \u2013 29 lessons - 2 hours 27 min\nLearn UI design\nIn this module, you'll learn how to build apps that both look good and work well.\nProjects: Intro to UI design, Build A Simple Layout, Full Project Refactoring\n\nModule 10 \u2013 167 lessons - 13 hours 3 min\nReact basics\nLearn the most popular library for building user interfaces. This will increase your productivity by an order of magnitude.",
        "metadata": {}
    },
    {
        "id": 33,
        "content": "Projects: Build a React info site, Build an AirBnb Experiences clone, Build a meme generator, Build a notes app and Tenzies games\n\nModule 11 \u2013 170 lessons - 13 hours 5 min\nAdvanced React [UPDATED]\nLevel up your React JS skills to a professional level.\n\nSections: Reusability, React Router 6, Performance",
        "metadata": {}
    },
    {
        "id": 34,
        "content": "Why the front end developer career path rocks:\nThis career path will turn you into a hireable frontend developer as fast as possible. By the end of it, you will have learned enough HTML, CSS, JavaScript, and React to get your first job as a frontend web developer.\n\nIt will also prepare you with strategies to get through the interview process, so that you increase the chance of landing your dream job.",
        "metadata": {}
    },
    {
        "id": 35,
        "content": "The teachers of this path are some of the most popular online instructors these days, like Kevin Powell, Gary Simon, Cassidy Williams, and Dylan Israel. They\u2019re all people who have gone up the hard road of becoming professional developers, so they know exactly what it takes.",
        "metadata": {}
    },
    {
        "id": 36,
        "content": "Throughout the path, you\u2019ll build more than a dozen projects, and solve more than 100 interactive coding challenges. In total, it clocks more than 70 hours. It\u2019s fully self-paced, and you can choose whether you\u2019d like to do it part-time or full-time.",
        "metadata": {}
    },
    {
        "id": 37,
        "content": "Career Path FAQs\nWhat topics are covered in the Career Path?\nThe Career Path aims to teach you everything you need to know to be hired as a Frontend developer. That includes HTML, CSS, JavaScript, React, UI Design, career advice and more! Check out the syllabus above to see everything the Career Path has to offer.",
        "metadata": {}
    },
    {
        "id": 38,
        "content": "Do I have to study full-time?\nThe Career Path, and indeed all Scrimba courses, is completely self-paced, so you can study full-time or part-time, alongside your other commitments. That said, we recommend that you study as often as possible to give yourself the best chance of progressing. Every day is great, if you can manage it.",
        "metadata": {}
    },
    {
        "id": 39,
        "content": "Where can I go for help?\nAt Scrimba, we believe that learning to code should be a community activity, which is why we've set up our Discord server - a place where you can meet fellow coders, share your code problems and solutions, and network. Just go to this link: https://scrimba.com/discord to sign up.",
        "metadata": {}
    },
    {
        "id": 40,
        "content": "Do I have to study the Career Path in the given order?\nAlthough we have set up the Career Path to allow you to go from zero knowledge to hireable developer, you don\u2019t have to follow the course in order we have given. You might choose to skip ahead to modules you need to know about more urgently, or go back and redo parts from time to time. It\u2019s your Career Path, so you can study however you like.",
        "metadata": {}
    },
    {
        "id": 41,
        "content": "I already know the content of a module, can I skip it?\nYou are free to skip modules that you already know, however we recommend that you try to complete all the challenges to test your knowledge before doing so. Please note that the certificate can only be issued when all the screencasts have been watched.",
        "metadata": {}
    },
    {
        "id": 42,
        "content": "What do I get when I complete the Career Path?\nWhen you complete the Career Path, you will automatically unlock the Frontend Developer Career Path certificate. You are also eligible for the \u2018Career Path Graduate\u2019 Discord badge. Just send a message to one of the team on Discord or email help@scrimba.com to claim this.",
        "metadata": {}
    },
    {
        "id": 43,
        "content": "Will the content of the Career Path be updated?\nYes, we are always open to feedback and constantly looking for ways to improve our courses. If you have any comments, please reach out to us on Discord or email help@scrimba.com.\n\nI completed the course and unlocked my certificate, but the Career Path has since been updated and my progress is no longer 100%.\nIf you have already completed the Career Path, we can fix up your progress so it shows 100% again. Just email help@scrimba.com.",
        "metadata": {}
    },
    {
        "id": 44,
        "content": "I have a question which isn\u2019t covered here.\nJust drop us a message on our Discord server (https://scrimba.com/discord) or email us at help@scrimba.com. We\u2019d be happy to help.\n\nGeneral Scrimba Info:\n\nPlans and Pricing:\nWe use purchasing power parity, so please check Scrimba.com for your local price.\n\nBuy with confidence\nMoney Back Guarantee:Full 30-day guarantee, no questions asked.\n\nKeep Discount For Life: Your discount stays for as long as you subscribe.",
        "metadata": {}
    },
    {
        "id": 45,
        "content": "Cancel Whenever: No lock-in period. Cancel whenever you want.\n\nBuy Scrimba as a Gift:\nOne-time purchase\nGive a membership without worrying about future payments\n\n30-day money-back guarantee\nNo questions asked, full refund within 24 hours\n\nDoes not expire\nOur gift vouchers do not expire\n\nAccess ALL our content\nTake all our courses, including the Frontend Developer Career Path\n\nEasy to use\nJust send a link and your recipient is good to go!",
        "metadata": {}
    },
    {
        "id": 46,
        "content": "Community\nJoin a global community of friendly and helpful coders\n\nDiscord Town Hall - ask team Scrimba anything. Hear back immediately! Live every Tuesday 5pm GMT/Midday EST. Various members of the Scrimba team smiling and Pumpkin the tuxedo cat saying 'Welcome' in a speech bubble.\nTown Hall",
        "metadata": {}
    },
    {
        "id": 47,
        "content": "A weekly podcast to keep you inspired\nOn The Scrimba Podcast, you'll hear motivational advice and job-hunting strategies from developers who've been exactly where you are now. Both from fresh success stories and seasoned experts.\n\nGet unstuck in minutes\nWe\u2019ve all been there. You\u2019re stuck with a bug you don\u2019t know how to fix, which makes your motivation slip. Thankfully you\u2019re not alone! Jump in the support channels and ask for a helping hand. You might just get unstuck in minutes!",
        "metadata": {}
    },
    {
        "id": 48,
        "content": "Find worldwide coding friends\nMeet like-minded peers on our Discord Server. Set goals and help each other to smash them through our accountabilty channels. Ask for help at any time of the day or night - there's always someone online.",
        "metadata": {}
    },
    {
        "id": 49,
        "content": "Power Hours to keep you focused\nA distraction-free, highly focused hour where you work on an important task that moves you forward. Power hours take place in the Scrimba Discord 4x a day in order to suit most time zones. Show up every weekday for a month, and you'll be shocked at how much progress you make!",
        "metadata": {}
    }
]
embeddings = get_1536_embeddings(sentences)
response = bulk_push_to_supabase(sentences, embeddings)

if response.data:
    print("Successfully inserted into Supabase.")
else:
    print("Failed to insert into Supabase:", response.data)
