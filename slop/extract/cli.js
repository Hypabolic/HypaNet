#!/usr/bin/env node
// Entry point for the UI-copy extractor. The sources are TypeScript (erasable syntax only), run by Node's built-in
// type stripping (Node 22.18 or later).
await import('./src/cli.ts');
