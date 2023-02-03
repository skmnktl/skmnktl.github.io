1. Build your schema first. Finalize the endpoints and their behaviors. That will minimize the work necessary.
2. Build the backend next.
    - What data do you need to store? How? Fix the schema, and then the database.
    - Authentication is of three dimensions
        - user to RESTful api,
        - api to databases, and
        - api to any services it calls.
    - Questions to ask:
        - Do any tasks need deferral to a backgroud job?
        - How will this be deployed? 
        - What exogenous environmental variables should serve as toggles in this system? 
        - Can we implement validation for the inputs into the api? 
        - For each function:
            - which errors are to be expected? 
 
3. For deployment,
    - 
