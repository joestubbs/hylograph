examples= [
    {
        "question": "How many space objects are in the catalog?",
        "query": "MATCH (n:SpaceObject) RETURN count(n) as nodes",
    },
    {
        "question": "How many orbit types are there in the catalog?",
        "query": "MATCH (o:OrbitType) RETURN count(o) as nodes",
    },
    {
        "question": "What are the names of the data sources that are private?",
        "query": "MATCH (d: DataSource {{PublicData: FALSE}}) RETURN d.Name"
    },
    {
        "question": "Tell me about the different orbit types that exist in the catalog?",
        "query": "MATCH (d:OrbitType) RETURN d",
    },
    {
        "question": "Which space object has the largest AreaToMass?",
        "query": "MATCH (d:SpaceObject) RETURN d ORDER BY d.AreaToMass DESC LIMIT 1", 
    }

]
