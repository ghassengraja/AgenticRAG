from redis import Redis
from langgraph.checkpoint.redis import RedisSaver




# We need to add a docker container for redis 
# docker run -d --name redis-demo -p 6379:6379 redis



client = Redis(
    host="localhost",  # or your Redis Enterprise endpoint
    port=6379,  # or 10000 for Azure Managed Redis / Azure Enterprise with TLS
    #password="your-access-key",
    #ssl=True,  # Azure/Enterprise typically requires SSL
    #ssl_cert_reqs="required",  # or "none" for self-signed certs
    #decode_responses=False  # RedisSaver expects bytes
)

# Pass the configured client to RedisSaver
saver = RedisSaver(redis_client=client)
saver.setup()

