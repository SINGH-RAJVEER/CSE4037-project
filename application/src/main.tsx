import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/query";
import ChessGame from "./routes/index";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ChessGame />
    </QueryClientProvider>
  </StrictMode>,
);
