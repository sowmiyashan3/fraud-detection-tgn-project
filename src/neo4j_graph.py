from neo4j import GraphDatabase

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_transaction_edges(self, edges_df):
        with self.driver.session() as session:
            for _, row in edges_df.iterrows():
                session.run("""
                    MERGE (src:Entity {name:$src})
                    MERGE (dst:Entity {name:$dst})
                    MERGE (src)-[r:INTERACTS {edge_type:$edge_type, timestamp:$ts, tgn_score:$score}]->(dst)
                """, src=row['src'], dst=row['dst'], edge_type=row['edge_type'],
                     ts=int(row['timestamp']), score=float(row['tgn_score']))

class FraudGraphStore:
    def __init__(self, uri, user, password):
        self.graph = Neo4jGraph(uri, user, password)
        self.driver = self.graph.driver

    def create_constraints(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def load_transaction_graph(self, edges_df, scores):
        edges_df = edges_df.copy()
        edges_df["tgn_score"] = scores
        self.graph.insert_transaction_edges(edges_df)

    def close(self):
        self.graph.close()