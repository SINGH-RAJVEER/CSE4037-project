FROM oven/bun:latest AS base
WORKDIR /app

# Install dependencies using workspace root lockfile
FROM base AS install
COPY package.json bun.lock ./
COPY application/package.json ./application/
COPY engine/package.json ./engine/
RUN bun install --frozen-lockfile

# Build React SPA
FROM base AS build
COPY --from=install /app/node_modules ./node_modules
COPY --from=install /app/application/node_modules ./application/node_modules
COPY package.json ./
COPY application/ ./application/
RUN cd application && bun run build

# Minimal runtime image
FROM base AS runner
WORKDIR /app
COPY --from=install /app/node_modules ./node_modules
COPY --from=install /app/application/node_modules ./application/node_modules
COPY package.json ./
COPY application/package.json ./application/
COPY application/src ./application/src
COPY application/drizzle ./application/drizzle
COPY --from=build /app/application/dist ./application/dist
EXPOSE 3000
WORKDIR /app/application
CMD ["bun", "run", "src/server.ts"]
