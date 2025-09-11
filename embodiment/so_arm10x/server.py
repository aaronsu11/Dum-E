from fastmcp import FastMCP, Context
import asyncio

mcp = FastMCP(
    name="SO-ARM10x MCP Server",
    instructions="This is a MCP server for the SO-ARM10x robot",
)


@mcp.tool(meta={"long_running": True})
async def dummy_long_running_tool(items: list[str], ctx: Context) -> dict:
    """For example, assign a task to the SO-ARM10x robot and report progress."""
    total = len(items)
    results = []

    for i, item in enumerate(items):
        # Report progress as we process each item
        await ctx.report_progress(
            progress=i,
            total=total,
            message=f"Assigning task to the SO-ARM10x robot: {item}",
        )

        # Simulate processing time
        await asyncio.sleep(5)
        results.append(item.upper())

    # Report 100% completion
    await ctx.report_progress(
        progress=total,
        total=total,
        message=f"Assigned {len(results)} tasks to the SO-ARM10x robot",
    )

    return {"assigned": len(results), "results": results}


@mcp.tool()
async def dummy_short_running_tool(item: str) -> dict:
    """For example, checking the status of a task."""
    await asyncio.sleep(5)
    return {"item": item, "status": "done"}


if __name__ == "__main__":
    mcp.run()
