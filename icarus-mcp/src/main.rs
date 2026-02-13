mod server;
mod tools;

use server::IcarusMcpServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mcp_core::run(IcarusMcpServer::new()).await
}
