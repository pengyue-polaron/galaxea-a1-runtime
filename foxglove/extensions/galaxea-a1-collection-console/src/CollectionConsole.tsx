import { MessageEvent, PanelExtensionContext } from "@foxglove/extension";
import {
  CSSProperties,
  ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
} from "react";
import { createRoot } from "react-dom/client";

import { COLLECTION_SERVICES, WORKFLOW_STATUS_TOPIC } from "./a1Config";

type Progress = {
  id: string;
  label: string;
  current: number;
  total: number | null;
  phase: string;
  detail: string;
};

type WorkflowStatus = {
  available: boolean;
  error: string;
  active: boolean;
  runId: string;
  state: string;
  workflow: string;
  name: string;
  inputPhase: string;
  inputDetail: string;
  inputActions: string[];
  progress: Progress[];
};

type ServiceName = keyof typeof COLLECTION_SERVICES;

type Control = {
  service: ServiceName;
  title: string;
  subtitle: string;
  tone: "primary" | "warning" | "danger" | "quiet";
  phase?: "ready" | "recording";
  confirm?: string;
};

const CONTROLS: Control[] = [
  {
    service: "start",
    title: "开始录制",
    subtitle: "Start recording",
    tone: "primary",
    phase: "ready",
  },
  {
    service: "save",
    title: "停止并保存",
    subtitle: "Stop & save",
    tone: "primary",
    phase: "recording",
  },
  {
    service: "reset",
    title: "回到采集起点",
    subtitle: "Reset position",
    tone: "warning",
    phase: "ready",
    confirm: "Reset the powered robot to the tracked collection start pose?",
  },
  {
    service: "discard",
    title: "放弃本条数据",
    subtitle: "Discard episode",
    tone: "danger",
    phase: "recording",
    confirm: "Discard the current episode without saving it?",
  },
  {
    service: "stop",
    title: "结束整个会话",
    subtitle: "End collection session",
    tone: "quiet",
    confirm: "Stop the active collection session?",
  },
];

const EMPTY_STATUS: WorkflowStatus = {
  available: false,
  error: "Waiting for workflow telemetry",
  active: false,
  runId: "",
  state: "idle",
  workflow: "",
  name: "",
  inputPhase: "",
  inputDetail: "",
  inputActions: [],
  progress: [],
};

function CollectionConsole({ context }: { context: PanelExtensionContext }): ReactElement {
  const [status, setStatus] = useState<WorkflowStatus>(EMPTY_STATUS);
  const [receivedAt, setReceivedAt] = useState(0);
  const [now, setNow] = useState(Date.now());
  const [busy, setBusy] = useState<ServiceName>();
  const [feedback, setFeedback] = useState("");
  const [feedbackError, setFeedbackError] = useState(false);
  const [colorScheme, setColorScheme] = useState<"dark" | "light">("dark");
  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();

  useLayoutEffect(() => {
    context.onRender = (renderState, done) => {
      if (renderState.colorScheme != undefined) {
        setColorScheme(renderState.colorScheme);
      }
      const event = latestStatusEvent(renderState.currentFrame);
      if (event != undefined) {
        try {
          setStatus(parseStatusMessage(event.message));
          setReceivedAt(Date.now());
        } catch (error) {
          setStatus({
            ...EMPTY_STATUS,
            error: error instanceof Error ? error.message : "Invalid workflow telemetry",
          });
          setReceivedAt(Date.now());
        }
      }
      setRenderDone(() => done);
    };
    context.watch("currentFrame");
    context.watch("colorScheme");
    context.subscribe([{ topic: WORKFLOW_STATUS_TOPIC }]);
    return () => {
      context.unsubscribeAll();
      context.onRender = undefined;
    };
  }, [context]);

  useEffect(() => {
    renderDone?.();
  }, [renderDone, status, colorScheme]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNow(Date.now());
    }, 500);
    return () => {
      window.clearInterval(timer);
    };
  }, []);

  const stale = receivedAt === 0 || now - receivedAt > 3000;
  const indicator = useMemo(() => statusIndicator(status, { stale }), [status, stale]);
  const collectionProgress = status.progress.find((item) => item.id === "collection");
  const captureProgress = status.progress.find((item) => item.id === "capture");
  const visibleProgress =
    status.inputPhase === "recording"
      ? (captureProgress ?? collectionProgress)
      : collectionProgress;
  const controlsAvailable = context.callService != undefined;

  const invoke = useCallback(
    async (control: Control) => {
      if (control.confirm != undefined && !window.confirm(control.confirm)) {
        return;
      }
      if (context.callService == undefined) {
        setFeedbackError(true);
        setFeedback("This Foxglove connection does not support service calls.");
        return;
      }
      setBusy(control.service);
      setFeedback("");
      setFeedbackError(false);
      try {
        const response = await context.callService(COLLECTION_SERVICES[control.service], {});
        const result = serviceResponse(response);
        if (!result.success) {
          throw new Error(result.message || `${control.subtitle} was rejected`);
        }
        setFeedback(result.message || `${control.subtitle} accepted`);
      } catch (error) {
        setFeedbackError(true);
        setFeedback(error instanceof Error ? error.message : `${control.subtitle} failed`);
      } finally {
        setBusy(undefined);
      }
    },
    [context],
  );

  const palette = colorScheme === "dark" ? DARK_PALETTE : LIGHT_PALETTE;
  const rootStyle: CSSProperties = {
    minHeight: "100%",
    boxSizing: "border-box",
    padding: 16,
    color: palette.text,
    background: palette.background,
    fontFamily: "Inter, system-ui, sans-serif",
  };

  return (
    <main style={rootStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          aria-label={indicator.label}
          style={{
            width: 12,
            height: 12,
            borderRadius: "50%",
            background: indicator.color,
            boxShadow: `0 0 10px ${indicator.color}`,
          }}
        />
        <div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{indicator.label}</div>
          <div style={{ color: palette.muted, fontSize: 12 }}>{indicator.detail}</div>
        </div>
      </div>

      {status.inputDetail && !stale ? (
        <div style={{ marginTop: 12, fontSize: 13, color: palette.text }}>
          {status.inputDetail}
        </div>
      ) : undefined}

      {visibleProgress != undefined && !stale ? (
        <section style={{ marginTop: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
            <span>{visibleProgress.label}</span>
            <span style={{ color: palette.muted }}>{progressValue(visibleProgress)}</span>
          </div>
          {visibleProgress.total != undefined ? (
            <div
              style={{
                height: 5,
                borderRadius: 3,
                marginTop: 5,
                overflow: "hidden",
                background: palette.border,
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${Math.min(100, (visibleProgress.current / visibleProgress.total) * 100)}%`,
                  background: indicator.color,
                }}
              />
            </div>
          ) : undefined}
          {visibleProgress.detail ? (
            <div style={{ color: palette.muted, fontSize: 11, marginTop: 4 }}>
              {visibleProgress.detail}
            </div>
          ) : undefined}
        </section>
      ) : undefined}

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: 8,
          marginTop: 14,
        }}
      >
        {CONTROLS.map((control) => {
          const enabled = controlEnabled(control, status, {
            stale,
            busy,
            servicesAvailable: controlsAvailable,
          });
          return (
            <button
              key={control.service}
              type="button"
              disabled={!enabled}
              onClick={() => void invoke(control)}
              style={buttonStyle(control.tone, { enabled, palette })}
            >
              <span style={{ display: "block", fontWeight: 700 }}>
                {busy === control.service ? "处理中…" : control.title}
              </span>
              <span style={{ display: "block", fontSize: 10, marginTop: 2, opacity: 0.78 }}>
                {control.subtitle}
              </span>
            </button>
          );
        })}
      </section>

      {feedback ? (
        <div
          role="status"
          style={{
            marginTop: 10,
            padding: "7px 9px",
            borderRadius: 5,
            fontSize: 11,
            color: feedbackError ? palette.danger : palette.success,
            background: palette.surface,
          }}
        >
          {feedback}
        </div>
      ) : undefined}

      <footer style={{ marginTop: 10, color: palette.muted, fontSize: 10 }}>
        {status.runId ? `run ${status.runId.slice(0, 8)} · ` : ""}
        {WORKFLOW_STATUS_TOPIC}
      </footer>
    </main>
  );
}

function latestStatusEvent(frame: readonly MessageEvent[] | undefined): MessageEvent | undefined {
  const matches = frame?.filter((event) => event.topic === WORKFLOW_STATUS_TOPIC);
  return matches?.[matches.length - 1];
}

function parseStatusMessage(message: unknown): WorkflowStatus {
  if (!isRecord(message) || typeof message.data !== "string") {
    throw new Error("Workflow status is not std_msgs/String");
  }
  const value: unknown = JSON.parse(message.data);
  if (!isRecord(value) || value.schema_version !== 2 || typeof value.available !== "boolean") {
    throw new Error("Workflow status schema mismatch");
  }
  if (!value.available) {
    return {
      ...EMPTY_STATUS,
      error: typeof value.error === "string" ? value.error : "Operator Session unavailable",
    };
  }
  const requiredStrings = [
    "run_id",
    "state",
    "workflow",
    "name",
    "input_phase",
    "input_detail",
  ] as const;
  if (
    typeof value.active !== "boolean" ||
    requiredStrings.some((key) => typeof value[key] !== "string") ||
    !Array.isArray(value.input_actions) ||
    !Array.isArray(value.progress)
  ) {
    throw new Error("Workflow status fields are invalid");
  }
  const inputActions = value.input_actions.map((item) => {
    if (!isRecord(item) || typeof item.id !== "string") {
      throw new Error("Workflow input action is invalid");
    }
    return item.id;
  });
  return {
    available: true,
    error: "",
    active: value.active,
    runId: value.run_id as string,
    state: value.state as string,
    workflow: value.workflow as string,
    name: value.name as string,
    inputPhase: value.input_phase as string,
    inputDetail: value.input_detail as string,
    inputActions,
    progress: value.progress.map(parseProgress),
  };
}

function parseProgress(value: unknown): Progress {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    typeof value.label !== "string" ||
    typeof value.current !== "number" ||
    !("total" in value) ||
    (value.total != null && typeof value.total !== "number") ||
    typeof value.phase !== "string" ||
    typeof value.detail !== "string"
  ) {
    throw new Error("Workflow progress is invalid");
  }
  return {
    id: value.id,
    label: value.label,
    current: value.current,
    total: typeof value.total === "number" ? value.total : null,
    phase: value.phase,
    detail: value.detail,
  };
}

function serviceResponse(value: unknown): { success: boolean; message: string } {
  if (!isRecord(value) || typeof value.success !== "boolean") {
    throw new Error("ROS Trigger returned an invalid response");
  }
  return {
    success: value.success,
    message: typeof value.message === "string" ? value.message : "",
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value != null && !Array.isArray(value);
}

function statusIndicator(
  status: WorkflowStatus,
  { stale }: { stale: boolean },
): { label: string; detail: string; color: string } {
  if (stale) {
    return { label: "遥测已断开 / Offline", detail: status.error, color: "#ef4444" };
  }
  if (!status.available) {
    return { label: "等待采集会话 / No session", detail: status.error, color: "#f59e0b" };
  }
  if (!status.active) {
    const label = status.state === "succeeded" ? "采集会话已完成 / Completed" : "空闲 / Idle";
    return { label, detail: status.name || "Open a collection from the terminal", color: "#64748b" };
  }
  if (status.inputPhase === "ready") {
    return { label: "等待开始录制 / Ready", detail: status.name, color: "#22c55e" };
  }
  if (status.inputPhase === "recording") {
    return { label: "正在录制 / Recording", detail: status.name, color: "#ef4444" };
  }
  const progressPhase =
    status.progress.find((item) => item.id === "collection")?.phase ??
    status.progress[status.progress.length - 1]?.phase;
  const labels: Record<string, string> = {
    saving: "正在保存 / Saving",
    discarding: "正在放弃 / Discarding",
    resetting: "机械臂复位中 / Resetting",
    completed: "采集已完成 / Completed",
  };
  return {
    label: labels[progressPhase ?? ""] ?? "处理中 / Busy",
    detail: status.name,
    color: status.state === "failed" ? "#ef4444" : "#3b82f6",
  };
}

function controlEnabled(
  control: Control,
  status: WorkflowStatus,
  {
    stale,
    busy,
    servicesAvailable,
  }: {
    stale: boolean;
    busy: ServiceName | undefined;
    servicesAvailable: boolean;
  },
): boolean {
  if (stale || busy != undefined || !servicesAvailable || !status.available || !status.active) {
    return false;
  }
  if (control.service === "stop") {
    return status.workflow === "collect" && status.state !== "stopping";
  }
  return (
    status.workflow === "collect" &&
    status.state === "waiting_for_input" &&
    status.inputPhase === control.phase &&
    status.inputActions.includes(control.service)
  );
}

function progressValue(progress: Progress): string {
  if (progress.total == undefined) {
    return String(progress.current);
  }
  return `${progress.current.toFixed(1)} / ${progress.total.toFixed(1)}`;
}

type Palette = typeof DARK_PALETTE;

const DARK_PALETTE = {
  background: "#111827",
  surface: "#1f2937",
  border: "#374151",
  text: "#f8fafc",
  muted: "#94a3b8",
  success: "#86efac",
  danger: "#fca5a5",
};

const LIGHT_PALETTE: Palette = {
  background: "#f8fafc",
  surface: "#ffffff",
  border: "#cbd5e1",
  text: "#0f172a",
  muted: "#64748b",
  success: "#15803d",
  danger: "#b91c1c",
};

function buttonStyle(
  tone: Control["tone"],
  { enabled, palette }: { enabled: boolean; palette: Palette },
): CSSProperties {
  const colors = {
    primary: { background: "#2563eb", color: "#ffffff", border: "#3b82f6" },
    warning: { background: "#b45309", color: "#ffffff", border: "#d97706" },
    danger: { background: "#b91c1c", color: "#ffffff", border: "#ef4444" },
    quiet: { background: palette.surface, color: palette.text, border: palette.border },
  }[tone];
  return {
    minHeight: 54,
    padding: "7px 9px",
    borderRadius: 6,
    border: `1px solid ${colors.border}`,
    background: colors.background,
    color: colors.color,
    cursor: enabled ? "pointer" : "not-allowed",
    opacity: enabled ? 1 : 0.38,
    textAlign: "left",
  };
}

export function initCollectionConsole(context: PanelExtensionContext): () => void {
  const root = createRoot(context.panelElement);
  root.render(<CollectionConsole context={context} />);
  return () => {
    root.unmount();
  };
}
