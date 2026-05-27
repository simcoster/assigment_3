import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as client:
        result = await client.call_tool("list_categories", {})
        print(result)

asyncio.run(main())