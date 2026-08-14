import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createHashRouter } from "react-router-dom";
import AppShell from "./components/AppShell";
import Dashboard from "./routes/Dashboard";
import PatientBrowser from "./routes/PatientBrowser";
import StudyViewer from "./routes/StudyViewer";
import Report from "./routes/Report";
import Audit from "./routes/Audit";
import Settings from "./routes/Settings";
import "./index.css";

// HashRouter: Tauri serves from a file:// origin where BrowserRouter breaks.
const router = createHashRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: "patients", element: <PatientBrowser /> },
      { path: "study/:id", element: <StudyViewer /> },
      { path: "report/:id", element: <Report /> },
      { path: "audit", element: <Audit /> },
      { path: "settings", element: <Settings /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
