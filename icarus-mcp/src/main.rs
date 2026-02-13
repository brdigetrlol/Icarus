// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

mod server;
mod tools;

use server::IcarusMcpServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mcp_core::run(IcarusMcpServer::new()).await
}
