import React, { useState, useCallback, useMemo } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { ThemeProvider, createTheme, alpha } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";

import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Chip,
  Switch,
  FormControlLabel,
  Divider,
} from "@mui/material";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
} from "recharts";

const API = process.env.REACT_APP_API || "http://localhost:8000";

/* ------------------------------------------------------------------ */
/*  DESIGN TOKENS                                                      */
/*  A "model diagnostics console" palette: deep ink-navy dark mode,    */
/*  teal (signal / correct) + violet (secondary) + rose (error) accents*/
/* ------------------------------------------------------------------ */

const getTokens = (darkMode) =>
  darkMode
    ? {
        bg: "#080C16",
        bgGradient:
          "radial-gradient(1200px 600px at 10% -10%, rgba(45,212,191,0.08), transparent), radial-gradient(1000px 500px at 100% 0%, rgba(167,139,250,0.08), transparent), #080C16",
        surface: "#101728",
        surfaceAlt: "#161F35",
        border: "rgba(148,163,209,0.14)",
        borderStrong: "rgba(148,163,209,0.26)",
        textPrimary: "#EAEFFB",
        textSecondary: "#8D9BC2",
        teal: "#2DD4BF",
        tealSoft: "rgba(45,212,191,0.16)",
        violet: "#A78BFA",
        violetSoft: "rgba(167,139,250,0.16)",
        amber: "#F5B942",
        rose: "#FB7185",
        roseSoft: "rgba(251,113,133,0.14)",
        tealText: "#5EEAD4",
        roseText: "#FDA4AF",
      }
    : {
        bg: "#F5F7FC",
        bgGradient:
          "radial-gradient(1200px 600px at 10% -10%, rgba(13,148,136,0.06), transparent), radial-gradient(1000px 500px at 100% 0%, rgba(124,92,252,0.06), transparent), #F5F7FC",
        surface: "#FFFFFF",
        surfaceAlt: "#F1F4FA",
        border: "#E3E8F3",
        borderStrong: "#D3DAEC",
        textPrimary: "#10162A",
        textSecondary: "#5B6785",
        teal: "#0D9488",
        tealSoft: "rgba(13,148,136,0.10)",
        violet: "#7C5CFC",
        violetSoft: "rgba(124,92,252,0.10)",
        amber: "#B45309",
        rose: "#E11D48",
        roseSoft: "rgba(225,29,72,0.08)",
        tealText: "#0D9488",
        roseText: "#E11D48",
      };

const FONT_DISPLAY = "'Space Grotesk', 'Segoe UI', sans-serif";
const FONT_BODY = "'Manrope', 'Segoe UI', sans-serif";
const FONT_MONO = "'JetBrains Mono', 'Roboto Mono', monospace";

/* ------------------------------------------------------------------ */
/*  CONFUSION MATRIX — the signature element                           */
/*  Full-width 2x2 heatmap with axis labels, monospace readouts        */
/* ------------------------------------------------------------------ */

function ConfusionMatrix({ matrix, classes, tokens }) {
  if (!matrix || matrix.length !== 2 || matrix[0].length !== 2) {
    return (
      <Typography sx={{ color: tokens.textSecondary, fontFamily: FONT_BODY }}>
        Confusion matrix unavailable for this run.
      </Typography>
    );
  }

  const labels =
    classes && classes.length === 2 ? classes : ["Class 0", "Class 1"];
  const total = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1];
  const maxVal = Math.max(
    matrix[0][0],
    matrix[0][1],
    matrix[1][0],
    matrix[1][1],
    1,
  );

  const Cell = ({ value, correct, rowLabel, colLabel }) => {
    const intensity = 0.18 + 0.72 * (value / maxVal);
    const base = correct ? tokens.teal : tokens.rose;
    const pct = total ? ((value / total) * 100).toFixed(1) : "0.0";

    return (
      <Box
        sx={{
          flex: 1,
          aspectRatio: "1 / 1",
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          borderRadius: { xs: "10px", sm: "14px" },
          border: `1px solid ${alpha(base, 0.4)}`,
          background: alpha(base, intensity * 0.35),
          boxShadow: correct
            ? `0 0 0 1px ${alpha(base, 0.15)} inset, 0 8px 24px ${alpha(base, 0.12)}`
            : `0 0 0 1px ${alpha(base, 0.1)} inset`,
          transition: "transform 0.2s ease, box-shadow 0.2s ease",
          cursor: "default",
          "&:hover": {
            transform: "translateY(-2px)",
            boxShadow: `0 12px 28px ${alpha(base, 0.22)}`,
          },
          p: { xs: 1, sm: 2 },
          textAlign: "center",
        }}
        title={`Actual: ${rowLabel} · Predicted: ${colLabel}`}
      >
        <Typography
          sx={{
            fontFamily: FONT_MONO,
            fontWeight: 700,
            fontSize: "clamp(1.6rem, 5vw, 3.4rem)",
            lineHeight: 1,
            color: correct ? tokens.tealText : tokens.roseText,
          }}
        >
          {value}
        </Typography>
        <Typography
          sx={{
            mt: 0.75,
            fontFamily: FONT_MONO,
            fontSize: "clamp(0.62rem, 1.1vw, 0.78rem)",
            letterSpacing: "0.08em",
            color: tokens.textSecondary,
          }}
        >
          {pct}%
        </Typography>
        <Typography
          sx={{
            mt: 0.25,
            fontFamily: FONT_BODY,
            fontWeight: 700,
            fontSize: "clamp(0.6rem, 1vw, 0.72rem)",
            letterSpacing: "0.04em",
            color: correct ? tokens.tealText : tokens.roseText,
            textTransform: "uppercase",
          }}
        >
          {correct ? "Correct" : "Missed"}
        </Typography>
      </Box>
    );
  };

  const axisLabelSx = {
    fontFamily: FONT_MONO,
    fontWeight: 600,
    fontSize: "clamp(0.68rem, 1.3vw, 0.85rem)",
    color: tokens.textSecondary,
    letterSpacing: "0.03em",
  };

  return (
    <Box sx={{ width: "100%", maxWidth: 980, mx: "auto" }}>
      {/* Predicted axis header */}
      <Box sx={{ display: "flex", mb: 1.5 }}>
        <Box sx={{ width: { xs: 64, sm: 96 }, flexShrink: 0 }} />
        <Box sx={{ flex: 1, textAlign: "center" }}>
          <Typography
            sx={{
              ...axisLabelSx,
              textTransform: "uppercase",
              color: tokens.violet,
            }}
          >
            ↓ Predicted Label
          </Typography>
        </Box>
      </Box>

      <Box sx={{ display: "flex" }}>
        {/* Actual axis (rotated) */}
        <Box
          sx={{
            width: { xs: 64, sm: 96 },
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Typography
            sx={{
              ...axisLabelSx,
              textTransform: "uppercase",
              color: tokens.violet,
              transform: "rotate(-90deg)",
              whiteSpace: "nowrap",
            }}
          >
            Actual Label →
          </Typography>
        </Box>

        <Box sx={{ flex: 1, minWidth: 0 }}>
          {/* column class headers */}
          <Box sx={{ display: "flex", mb: 1 }}>
            <Box sx={{ width: { xs: 64, sm: 96 }, flexShrink: 0 }} />
            {labels.map((l) => (
              <Typography
                key={l}
                sx={{ ...axisLabelSx, flex: 1, textAlign: "center" }}
              >
                {l}
              </Typography>
            ))}
          </Box>

          {/* row 0 */}
          <Box
            sx={{
              display: "flex",
              alignItems: "stretch",
              gap: { xs: 1, sm: 1.5 },
              mb: { xs: 1, sm: 1.5 },
            }}
          >
            <Box
              sx={{
                width: { xs: 64, sm: 96 },
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "flex-end",
                pr: 1,
              }}
            >
              <Typography sx={{ ...axisLabelSx, textAlign: "right" }}>
                {labels[0]}
              </Typography>
            </Box>
            <Cell
              value={matrix[0][0]}
              correct
              rowLabel={labels[0]}
              colLabel={labels[0]}
            />
            <Cell
              value={matrix[0][1]}
              correct={false}
              rowLabel={labels[0]}
              colLabel={labels[1]}
            />
          </Box>

          {/* row 1 */}
          <Box
            sx={{
              display: "flex",
              alignItems: "stretch",
              gap: { xs: 1, sm: 1.5 },
            }}
          >
            <Box
              sx={{
                width: { xs: 64, sm: 96 },
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "flex-end",
                pr: 1,
              }}
            >
              <Typography sx={{ ...axisLabelSx, textAlign: "right" }}>
                {labels[1]}
              </Typography>
            </Box>
            <Cell
              value={matrix[1][0]}
              correct={false}
              rowLabel={labels[1]}
              colLabel={labels[0]}
            />
            <Cell
              value={matrix[1][1]}
              correct
              rowLabel={labels[1]}
              colLabel={labels[1]}
            />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

/* ------------------------------------------------------------------ */
/*  APP                                                                 */
/* ------------------------------------------------------------------ */

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const tokens = useMemo(() => getTokens(darkMode), [darkMode]);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? "dark" : "light",
          background: { default: tokens.bg, paper: tokens.surface },
          primary: { main: tokens.teal },
          secondary: { main: tokens.violet },
          success: { main: tokens.teal },
          error: { main: tokens.rose },
          warning: { main: tokens.amber },
          text: {
            primary: tokens.textPrimary,
            secondary: tokens.textSecondary,
          },
          divider: tokens.border,
        },
        shape: { borderRadius: 16 },
        typography: {
          fontFamily: FONT_BODY,
          h3: {
            fontFamily: FONT_DISPLAY,
            fontWeight: 700,
            letterSpacing: "-0.02em",
          },
          h4: { fontFamily: FONT_DISPLAY, fontWeight: 700 },
          h5: { fontFamily: FONT_DISPLAY, fontWeight: 700 },
          h6: {
            fontFamily: FONT_BODY,
            fontWeight: 700,
            color: tokens.textSecondary,
          },
          button: {
            fontFamily: FONT_BODY,
            fontWeight: 700,
            textTransform: "none",
          },
        },
        components: {
          MuiCssBaseline: {
            styleOverrides: {
              body: {
                background: tokens.bgGradient,
                backgroundAttachment: "fixed",
                transition: "background 0.25s ease",
              },
              "::selection": { background: alpha(tokens.teal, 0.35) },
              "*:focus-visible": {
                outline: `2px solid ${tokens.teal}`,
                outlineOffset: "2px",
              },
            },
          },
          MuiCard: {
            styleOverrides: {
              root: {
                backgroundColor: alpha(tokens.surface, darkMode ? 0.72 : 1),
                backgroundImage: "none",
                border: `1px solid ${tokens.border}`,
                borderRadius: 20,
                backdropFilter: "blur(14px)",
                boxShadow: darkMode
                  ? "0 1px 0 rgba(255,255,255,0.03) inset, 0 20px 40px -24px rgba(0,0,0,0.6)"
                  : "0 1px 0 rgba(255,255,255,0.6) inset, 0 20px 40px -28px rgba(20,30,60,0.18)",
              },
            },
          },
          MuiButton: {
            styleOverrides: {
              containedPrimary: {
                background: `linear-gradient(135deg, ${tokens.teal}, ${tokens.violet})`,
                color: "#04120F",
                boxShadow: `0 8px 20px ${alpha(tokens.teal, 0.28)}`,
                "&:hover": {
                  boxShadow: `0 10px 26px ${alpha(tokens.teal, 0.4)}`,
                  filter: "brightness(1.05)",
                },
              },
              outlined: {
                borderColor: tokens.borderStrong,
                color: tokens.textPrimary,
              },
              root: { borderRadius: 12 },
            },
          },
          MuiChip: {
            styleOverrides: { root: { fontWeight: 700, borderRadius: 8 } },
          },
          MuiTextField: {
            styleOverrides: {
              root: {
                "& .MuiOutlinedInput-root": {
                  borderRadius: 12,
                  backgroundColor: alpha(
                    tokens.surfaceAlt,
                    darkMode ? 0.5 : 0.6,
                  ),
                  "& fieldset": { borderColor: tokens.border },
                  "&:hover fieldset": { borderColor: tokens.teal },
                  "&.Mui-focused fieldset": { borderColor: tokens.teal },
                },
                "& .MuiInputLabel-root": { color: tokens.textSecondary },
              },
            },
          },
          MuiAlert: {
            styleOverrides: { root: { borderRadius: 12 } },
          },
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [darkMode],
  );

  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [data, setData] = useState(null);

  const [predictionInputs, setPredictionInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [probability, setProbability] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length) {
      setFile(acceptedFiles[0]);
      setError("");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    multiple: false,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
    },
    onDrop,
  });

  const trainModel = async () => {
    if (!file) {
      setError("Please upload a dataset.");
      return;
    }

    if (!target) {
      setError("Enter target column.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const formData = new FormData();

      formData.append("file", file);
      formData.append("target_column", target);
      formData.append("task", "classification");

      const res = await axios.post(`${API}/train`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setData(res.data);

      const initial = {};

      if (res.data.columns && Array.isArray(res.data.columns)) {
        res.data.columns.forEach((c) => {
          if (c !== target) initial[c] = "";
        });
      }

      setPredictionInputs(initial);
    } catch (err) {
      console.log(err);
      const detail = err.response?.data?.detail;

      if (Array.isArray(detail)) {
        setError(detail[0].msg);
      } else {
        setError(detail || "Training failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePredictionInput = (key, value) => {
    setPredictionInputs((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const predict = async () => {
    try {
      const res = await axios.post(`${API}/predict`, predictionInputs);

      setPredictionResult(res.data.prediction);

      setProbability(res.data.probability);
    } catch (err) {
      console.log(err);
      alert("Prediction Failed");
    }
  };

  const leaderboard = data?.leaderboard || [];

  const featureImportance = data?.feature_importance
    ? Object.entries(data.feature_importance).map(([feature, value]) => ({
        feature,
        value,
      }))
    : [];

  const rocData = data?.roc_curve
    ? data.roc_curve.fpr.map((fpr, index) => ({
        fpr,
        tpr: data.roc_curve.tpr[index],
      }))
    : [];

  const metrics = data?.metrics || {};

  const sectionTitleSx = {
    fontSize: { xs: "1.1rem", sm: "1.3rem" },
    mb: { xs: 2, sm: 3 },
    color: tokens.textPrimary,
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: "100vh",
          background: tokens.bgGradient,
          px: { xs: 2, sm: 3, md: 5 },
          py: { xs: 3, md: 4 },
        }}
      >
        <Box sx={{ width: "100%" }}>
          {/* HEADER */}
          <Box
            sx={{
              display: "flex",
              flexDirection: { xs: "column", sm: "row" },
              justifyContent: "space-between",
              alignItems: { xs: "flex-start", sm: "center" },
              gap: 2,
              mb: { xs: 3, md: 5 },
            }}
          >
            <Box>
              <Typography
                variant="h3"
                sx={{
                  fontSize: { xs: "1.9rem", sm: "2.4rem", md: "2.8rem" },
                  background: `linear-gradient(90deg, ${tokens.teal}, ${tokens.violet})`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}
              >
                AutoML Studio
              </Typography>

              <Typography
                sx={{
                  color: tokens.textSecondary,
                  fontFamily: FONT_MONO,
                  fontSize: { xs: "0.8rem", sm: "0.9rem" },
                  letterSpacing: "0.04em",
                  mt: 0.5,
                }}
              >
                train · compare · explain · predict
              </Typography>
            </Box>

            <FormControlLabel
              sx={{ m: 0 }}
              control={
                <Switch
                  checked={darkMode}
                  onChange={() => setDarkMode(!darkMode)}
                  sx={{
                    "& .MuiSwitch-switchBase.Mui-checked": {
                      color: tokens.teal,
                    },
                    "& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track": {
                      backgroundColor: tokens.teal,
                    },
                  }}
                />
              }
              label={
                <Typography
                  sx={{ color: tokens.textSecondary, fontWeight: 600 }}
                >
                  Dark mode
                </Typography>
              }
            />
          </Box>

          {/* Upload */}
          <Card sx={{ mb: 4 }}>
            <CardContent sx={{ p: { xs: 2.5, sm: 4 } }}>
              <Typography variant="h5" sx={sectionTitleSx}>
                Upload dataset
              </Typography>

              <Box
                {...getRootProps()}
                sx={{
                  border: `2px dashed ${isDragActive ? tokens.teal : tokens.borderStrong}`,
                  borderRadius: "16px",
                  p: { xs: 3, sm: 5 },
                  textAlign: "center",
                  cursor: "pointer",
                  background: isDragActive
                    ? tokens.tealSoft
                    : alpha(tokens.surfaceAlt, 0.4),
                  transition: "all 0.2s ease",
                }}
              >
                <input {...getInputProps()} />

                {file ? (
                  <>
                    <Typography variant="h6" sx={{ color: tokens.textPrimary }}>
                      {file.name}
                    </Typography>

                    <Typography
                      sx={{ color: tokens.teal, mt: 0.5, fontWeight: 600 }}
                    >
                      Ready for training
                    </Typography>
                  </>
                ) : (
                  <>
                    <Typography variant="h6" sx={{ color: tokens.textPrimary }}>
                      Drag & drop CSV / Excel
                    </Typography>

                    <Typography sx={{ color: tokens.textSecondary, mt: 0.5 }}>
                      or click to browse
                    </Typography>
                  </>
                )}
              </Box>

              <Box
                sx={{
                  display: "flex",
                  flexDirection: { xs: "column", md: "row" },
                  gap: 2,
                  mt: 0.5,
                }}
              >
                <Box sx={{ flex: { md: 2 } }}>
                  <TextField
                    fullWidth
                    label="Target column"
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                  />
                </Box>

                <Box sx={{ flex: { md: 1 } }}>
                  <Button
                    fullWidth
                    variant="contained"
                    size="large"
                    sx={{ height: "56px" }}
                    onClick={trainModel}
                    disabled={loading}
                  >
                    {loading ? "Training…" : "Train model"}
                  </Button>
                </Box>
              </Box>

              {loading && (
                <Box mt={3} display="flex" alignItems="center" gap={2}>
                  <CircularProgress size={22} sx={{ color: tokens.teal }} />
                  <Typography sx={{ color: tokens.textSecondary }}>
                    Training in progress…
                  </Typography>
                </Box>
              )}

              {error && (
                <Alert sx={{ mt: 3 }} severity="error">
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>

          {data && (
            <>
              {/* Dashboard Cards */}
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "repeat(2, 1fr)",
                    sm: "repeat(3, 1fr)",
                    md: "repeat(6, 1fr)",
                  },
                  gap: 2,
                  mb: 4,
                }}
              >
                {[
                  ["Task", data.task],
                  ["Best model", data.best_model, true],
                  ["Score", `${(data.best_score * 100).toFixed(2)}%`],
                  ["Rows", data.dataset_rows],
                  ["Columns", data.dataset_columns],
                  ["Training", `${data.training_time}s`],
                ].map(([label, value, isChip], i) => (
                  <Card key={i} sx={{ height: "100%" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography
                        variant="h6"
                        sx={{
                          fontSize: "0.75rem",
                          textTransform: "uppercase",
                          letterSpacing: "0.06em",
                        }}
                      >
                        {label}
                      </Typography>

                      {isChip ? (
                        <Chip
                          label={value}
                          sx={{
                            mt: 1,
                            background: tokens.tealSoft,
                            color: tokens.tealText,
                          }}
                        />
                      ) : (
                        <Typography
                          sx={{
                            mt: 0.5,
                            fontFamily: FONT_MONO,
                            fontWeight: 700,
                            fontSize: { xs: "1.3rem", sm: "1.6rem" },
                            color: tokens.textPrimary,
                          }}
                        >
                          {value}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </Box>

              {/* Leaderboard + Metrics */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: { xs: "column", md: "row" },
                  gap: 3,
                  mb: 4,
                  alignItems: "stretch",
                }}
              >
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Card sx={{ height: "100%" }}>
                    <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                      <Typography variant="h5" sx={sectionTitleSx}>
                        Leaderboard
                      </Typography>

                      {leaderboard.map((model, index) => (
                        <Box
                          key={index}
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            p: 2,
                            mb: 1.5,
                            borderRadius: "12px",
                            border: `1px solid ${index === 0 ? alpha(tokens.teal, 0.35) : tokens.border}`,
                            background:
                              index === 0
                                ? tokens.tealSoft
                                : alpha(tokens.surfaceAlt, 0.5),
                          }}
                        >
                          <Box>
                            <Typography
                              sx={{
                                fontWeight: 700,
                                color: tokens.textPrimary,
                              }}
                            >
                              {index + 1}. {model.model}
                            </Typography>

                            <Typography
                              sx={{
                                color: tokens.textSecondary,
                                fontFamily: FONT_MONO,
                                fontSize: "0.85rem",
                              }}
                            >
                              {(model.score * 100).toFixed(2)}%
                            </Typography>
                          </Box>

                          <Chip
                            label={index === 0 ? "Best" : "Candidate"}
                            sx={{
                              background:
                                index === 0
                                  ? tokens.tealSoft
                                  : tokens.violetSoft,
                              color:
                                index === 0 ? tokens.tealText : tokens.violet,
                            }}
                          />
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Box>

                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Card sx={{ height: "100%" }}>
                    <CardContent
                      sx={{
                        p: { xs: 2.5, sm: 3.5 },
                        height: "100%",
                        display: "flex",
                        flexDirection: "column",
                      }}
                    >
                      <Typography variant="h5" sx={sectionTitleSx}>
                        Metrics
                      </Typography>

                      <Box
                        sx={{
                          flex: 1,
                          display: "flex",
                          flexDirection: "column",
                          gap: 1.5,
                        }}
                      >
                        {Object.entries(metrics).map(([key, value]) => (
                          <Box
                            key={key}
                            sx={{
                              flex: 1,
                              display: "flex",
                              flexDirection: "column",
                              justifyContent: "center",
                              gap: 0.75,
                              px: 3,
                              borderRadius: "12px",
                              border: `1px solid ${tokens.border}`,
                              background: alpha(tokens.surfaceAlt, 0.5),
                            }}
                          >
                            <Typography
                              sx={{
                                color: tokens.textSecondary,
                                fontSize: "0.78rem",
                                letterSpacing: "0.08em",
                                textTransform: "uppercase",
                                fontWeight: 700,
                              }}
                            >
                              {key.toUpperCase()}
                            </Typography>

                            <Typography
                              sx={{
                                fontFamily: FONT_MONO,
                                fontWeight: 700,
                                fontSize: { xs: "1.7rem", sm: "2.1rem" },
                                color: tokens.textPrimary,
                                lineHeight: 1,
                              }}
                            >
                              {typeof value === "number"
                                ? value.toFixed(4)
                                : value}
                            </Typography>
                          </Box>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              </Box>

              {/* CONFUSION MATRIX — full width, 2x2, signature element */}
              {data.confusion_matrix && (
                <Card sx={{ mb: 4 }}>
                  <CardContent sx={{ p: { xs: 2.5, sm: 4, md: 5 } }}>
                    <Typography variant="h5" sx={sectionTitleSx}>
                      Confusion matrix
                    </Typography>
                    <Typography
                      sx={{
                        color: tokens.textSecondary,
                        mb: { xs: 3, md: 4 },
                        fontSize: "0.9rem",
                      }}
                    >
                      How predictions compare to actual outcomes across both
                      classes.
                    </Typography>

                    <ConfusionMatrix
                      matrix={data.confusion_matrix}
                      classes={data.classes}
                      tokens={tokens}
                    />
                  </CardContent>
                </Card>
              )}

              {/* Feature Importance */}
              <Card sx={{ mb: 4 }}>
                <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                  <Typography variant="h5" sx={sectionTitleSx}>
                    Feature importance
                  </Typography>

                  <ResponsiveContainer width="100%" height={420}>
                    <BarChart layout="vertical" data={featureImportance}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke={tokens.border}
                      />
                      <XAxis
                        type="number"
                        stroke={tokens.textSecondary}
                        tick={{ fill: tokens.textSecondary, fontSize: 12 }}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        stroke={tokens.textSecondary}
                        tick={{ fill: tokens.textSecondary, fontSize: 12 }}
                        width={110}
                      />
                      <Tooltip
                        contentStyle={{
                          background: tokens.surface,
                          border: `1px solid ${tokens.border}`,
                          borderRadius: 10,
                          color: tokens.textPrimary,
                        }}
                      />
                      <Bar
                        dataKey="value"
                        fill={tokens.teal}
                        radius={[0, 6, 6, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* ROC */}
              {rocData.length > 0 && (
                <Card sx={{ mb: 4 }}>
                  <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                    <Typography variant="h5" sx={sectionTitleSx}>
                      ROC curve
                    </Typography>

                    <ResponsiveContainer width="100%" height={380}>
                      <AreaChart data={rocData}>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke={tokens.border}
                        />
                        <XAxis
                          dataKey="fpr"
                          stroke={tokens.textSecondary}
                          tick={{ fill: tokens.textSecondary, fontSize: 12 }}
                        />
                        <YAxis
                          stroke={tokens.textSecondary}
                          tick={{ fill: tokens.textSecondary, fontSize: 12 }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: tokens.surface,
                            border: `1px solid ${tokens.border}`,
                            borderRadius: 10,
                            color: tokens.textPrimary,
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="tpr"
                          fill={alpha(tokens.violet, 0.25)}
                          stroke="none"
                        />
                        <Line
                          type="monotone"
                          dataKey="tpr"
                          stroke={tokens.violet}
                          strokeWidth={3}
                          dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* AI Insights + Bias Report */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: { xs: "column", md: "row" },
                  gap: 3,
                  mb: 4,
                  alignItems: "stretch",
                }}
              >
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Card sx={{ height: "100%" }}>
                    <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                      <Typography variant="h5" sx={sectionTitleSx}>
                        AI insights
                      </Typography>

                      <Divider sx={{ mb: 2, borderColor: tokens.border }} />

                      {data.insights && data.insights.length > 0 ? (
                        data.insights.map((item, index) => (
                          <Typography
                            key={index}
                            sx={{
                              mb: 2,
                              color: tokens.textPrimary,
                              "&::before": {
                                content: '"→ "',
                                color: tokens.teal,
                              },
                            }}
                          >
                            {item}
                          </Typography>
                        ))
                      ) : (
                        <Typography sx={{ color: tokens.textSecondary }}>
                          No insights generated.
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Box>

                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Card sx={{ height: "100%" }}>
                    <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                      <Typography variant="h5" sx={sectionTitleSx}>
                        Bias report
                      </Typography>

                      <Divider sx={{ mb: 2, borderColor: tokens.border }} />

                      {data.bias_report ? (
                        Object.entries(data.bias_report).map(([key, value]) => {
                          const ok = Math.abs(Number(value)) < 0.1;
                          return (
                            <Box
                              key={key}
                              sx={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                mb: 1.5,
                                p: 2,
                                borderRadius: "12px",
                                border: `1px solid ${tokens.border}`,
                                background: alpha(tokens.surfaceAlt, 0.5),
                              }}
                            >
                              <Typography sx={{ color: tokens.textPrimary }}>
                                {key}
                              </Typography>

                              <Chip
                                label={String(value)}
                                sx={{
                                  background: ok
                                    ? tokens.tealSoft
                                    : alpha(tokens.amber, 0.16),
                                  color: ok ? tokens.tealText : tokens.amber,
                                }}
                              />
                            </Box>
                          );
                        })
                      ) : (
                        <Typography sx={{ color: tokens.textSecondary }}>
                          Bias report unavailable.
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Box>
              </Box>

              {/* Download Buttons */}
              <Card sx={{ mb: 4 }}>
                <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                  <Typography variant="h5" sx={sectionTitleSx}>
                    Downloads
                  </Typography>

                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
                    <Button variant="contained" href={`${API}/download-model`}>
                      Download model
                    </Button>

                    <Button variant="outlined" href={`${API}/download-report`}>
                      Download report
                    </Button>
                  </Box>
                </CardContent>
              </Card>

              {/* Dynamic Prediction */}
              <Card sx={{ mb: 4 }}>
                <CardContent sx={{ p: { xs: 2.5, sm: 3.5 } }}>
                  <Typography variant="h5" sx={sectionTitleSx}>
                    Make a prediction
                  </Typography>

                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: {
                        xs: "1fr",
                        sm: "repeat(2, 1fr)",
                        md: "repeat(3, 1fr)",
                      },
                      gap: 2,
                    }}
                  >
                    {Object.keys(predictionInputs).map((column) => (
                      <TextField
                        key={column}
                        fullWidth
                        label={column}
                        value={predictionInputs[column]}
                        onChange={(e) =>
                          handlePredictionInput(column, e.target.value)
                        }
                      />
                    ))}
                  </Box>

                  <Button sx={{ mt: 3 }} variant="contained" onClick={predict}>
                    Predict
                  </Button>

                  {predictionResult !== null && (
                    <Box
                      sx={{
                        mt: 4,
                        p: 3,
                        borderRadius: "16px",
                        border: `1px solid ${predictionResult === 1 ? alpha(tokens.teal, 0.35) : alpha(tokens.rose, 0.35)}`,
                        background:
                          predictionResult === 1
                            ? tokens.tealSoft
                            : tokens.roseSoft,
                      }}
                    >
                      <Typography
                        variant="h4"
                        sx={{
                          fontSize: { xs: "1.3rem", sm: "1.6rem" },
                          color:
                            predictionResult === 1
                              ? tokens.tealText
                              : tokens.roseText,
                        }}
                      >
                        {predictionResult === 1
                          ? "Positive prediction"
                          : "Negative prediction"}
                      </Typography>

                      {probability !== null && (
                        <>
                          <Typography
                            mt={2}
                            sx={{
                              color: tokens.textSecondary,
                              fontFamily: FONT_MONO,
                            }}
                          >
                            Confidence: {(probability * 100).toFixed(2)}%
                          </Typography>

                          <Box
                            sx={{
                              mt: 2,
                              height: 10,
                              borderRadius: 10,
                              overflow: "hidden",
                              background: tokens.border,
                            }}
                          >
                            <Box
                              sx={{
                                width: `${probability * 100}%`,
                                height: "100%",
                                background: `linear-gradient(90deg, ${tokens.teal}, ${tokens.violet})`,
                              }}
                            />
                          </Box>
                        </>
                      )}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
