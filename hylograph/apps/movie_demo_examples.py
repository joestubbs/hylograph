examples = [
    {
        "question": "How many movies were released in 1995?",
        "query": "MATCH (m:Movie) WHERE m.year = 1995 RETURN count(*) AS result",
    },
    {
        "question": "Who directed the movie Inception?",
        "query": "MATCH (m:Movie {{title: 'Inception'}})<-[:DIRECTED]-(d) RETURN d.name",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Actor {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {{title: 'Schindler\\'s List'}})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "List movies that have an IMDb rating above 8.0 and have grossed over 100 million dollars.",
        "query": "MATCH (m:Movie) WHERE m.imdbRating > 8.0 AND m.revenue > 100000000 RETURN m.title, m.imdbRating, m.revenue",
    },
    {
        "question": "Find the top 3 movies with the highest budget in 1995.",
        "query": "MATCH (m:Movie) WHERE m.year = 1995 RETURN m.title, m.budget ORDER BY m.budget DESC LIMIT 3",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Director)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Actor) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "What is the total box-office revenue of all movies released in the 'Fantasy' genre?",
        "query": "MATCH (:Genre {{name: 'Fantasy'}})<-[:IN_GENRE]-(m:Movie) RETURN SUM(m.revenue) AS TotalRevenue",
    },
    {
        "question": "List all users who rated the same movie more than once.",
        "query": "MATCH (u:User)-[r:RATED]->(m:Movie) WITH u, m, COUNT(r) AS ratingsCount WHERE ratingsCount > 1 RETURN u.name, m.title, ratingsCount",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
    {
        "question": "Which movies have a plot that mentions 'revenge' and were released after 2010?",
        "query": "MATCH (m:Movie) WHERE m.plot CONTAINS 'revenge' AND m.year > 2010 RETURN m.title, m.released",
    },
    {
        "question": "How many movies are there in each genre?",
        "query": "MATCH (g:Genre)<-[:IN_GENRE]-(m:Movie) RETURN g.name, COUNT(m) AS movieCount",
    },
    {
        "question": "What are the names of actors born in 'Dallas, Texas, USA' who have acted in Mystery movies?",
        "query": "MATCH (a:Actor {{bornIn: 'London'}})-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(:Genre {{name: 'Mystery'}}) RETURN DISTINCT a.name",
    },
    {
        "question": "Identify the movie with the longest runtime that was released in the 2000s.",
        "query": "MATCH (m:Movie) WHERE m.year >= 2000 AND m.year < 2010 RETURN m.title, m.runtime ORDER BY m.runtime DESC LIMIT 1",
    },
    {
        "question": "List the languages available in movies directed by 'Steven Spielberg'.",
        "query": "MATCH (d:Director {{name: 'Steven Spielberg'}})-[:DIRECTED]->(m:Movie) UNWIND m.languages AS language RETURN DISTINCT language",
    },
    {
        "question": "Which movie had the highest IMDb rating in the year Quentin Tarantino was born?",
        "query": "MATCH (d:Director {{name: 'Quentin Tarantino'}}), (m:Movie) WHERE m.year = d.born.year RETURN m.title, m.imdbRating ORDER BY m.imdbRating DESC LIMIT 1",
    },
    {
        "question": "What are the most common genres for movies with a budget over 200 million dollars?",
        "query": "MATCH (m:Movie)-[:IN_GENRE]->(g:Genre) WHERE m.budget > 200000000 RETURN g.name, COUNT(*) AS genreCount ORDER BY genreCount DESC",
    },
    {
        "question": "Which countries are represented by at least five movies in the database?",
        "query": "MATCH (m:Movie) UNWIND m.countries AS country WITH country, COUNT(m) AS movieCount WHERE movieCount >= 5 RETURN country, movieCount",
    },
    {
        "question": "List all movies that involve time travel in their plot.",
        "query": "MATCH (m:Movie) WHERE m.plot CONTAINS 'time travel' RETURN m.title",
    },
    {
        "question": "What are the average IMDb ratings for each year in the 21st century?",
        "query": "MATCH (m:Movie) WHERE m.year >= 2000 AND m.year < 2100 WITH m.year AS year, AVG(m.imdbRating) AS avgRating RETURN year, avgRating ORDER BY year",
    },
    {
        "question": "List top 10 actors with the most diverse genres in their filmography.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g:Genre) WITH a, COUNT(DISTINCT g) AS genreDiversity ORDER BY genreDiversity DESC LIMIT 10 RETURN a.name, genreDiversity",
    },
    {
        "question": "Which directors have never had a movie with a rating below 6.0?",
        "query": "MATCH (d:Director)-[:DIRECTED]->(m:Movie) WITH d, MIN(m.imdbRating) AS lowestRating WHERE lowestRating >= 6.0 RETURN d.name, lowestRating",
    },
    {
        "question": "How many movies have the keyword 'love' in the title and a runtime under 2 hours?",
        "query": "MATCH (m:Movie) WHERE m.title CONTAINS 'love' AND m.runtime < 120 RETURN COUNT(m) AS numberOfMovies",
    },
    {
        "question": "Return the list of movies that have a higher IMDb rating than any of 'Tom Hanks' movies.",
        "query": "MATCH (m:Movie), (th:Actor {{name: 'Tom Hanks'}})-[:ACTED_IN]->(tomHanksMovie:Movie) WITH MAX(tomHanksMovie.imdbRating) AS maxTomHanksRating MATCH (m) WHERE m.imdbRating > maxTomHanksRating RETURN m.title, m.imdbRating",
    },
]
