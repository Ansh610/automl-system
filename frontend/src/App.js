import React, { useState, useCallback } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { ThemeProvider, createTheme } from "@mui/material/styles";

import {
  Box,
  Grid,
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
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
  Area,
} from "recharts";

const API = "https://automl-system-1.onrender.com";

function App() {
  const [darkMode, setDarkMode] = useState(false);

  const theme = createTheme({
    palette: { mode: darkMode ? "dark" : "light" },
  });

  const textColor = darkMode ? "#fff" : "#000";

  const tooltipStyle = {
    backgroundColor: darkMode ? "#1e293b" : "#fff",
    color: darkMode ? "#fff" : "#000",
  };

  const [data, setData] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    setFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    accept: { "text/csv": [".csv"] },
    onDrop,
  });

  const trainModel = async () => {
    if (!file) {
      alert("Upload dataset first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(`${API}/train?target=converted`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setData(res.data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Training failed");
    } finally {
      setLoading(false);
    }
  };

  const predict = async () => {
    const input = {
      age: Number(document.getElementById("age").value),
      income: Number(document.getElementById("income").value),
      city: document.getElementById("city").value,
      gender: document.getElementById("gender").value,
      website_visits: Number(document.getElementById("visits").value),
      time_spent: Number(document.getElementById("time").value),
    };

    try {
      const res = await axios.post(`${API}/predict`, input);

      setPrediction(res.data.prediction);
      setProbability(res.data.probability);
    } catch (err) {
      console.error("Prediction failed", err);
    }
  };

  const modelScores = data
    ? Object.entries(data.model_scores).map(([model, score]) => ({
        model,
        score,
      }))
    : [];

  const featureImportance = data
    ? Object.entries(data.feature_importance).map(([feature, value]) => ({
        feature,
        value,
      }))
    : [];

  const metricsChart = data
    ? [
        { metric: "Accuracy", value: data.model_metrics?.accuracy },
        { metric: "Precision", value: data.model_metrics?.precision },
        { metric: "Recall", value: data.model_metrics?.recall },
        { metric: "F1 Score", value: data.model_metrics?.f1 },
      ]
    : [];

  const rocData =
    data && data.roc_curve
      ? data.roc_curve.fpr.map((v, i) => ({
          fpr: v,
          tpr: data.roc_curve.tpr[i],
        }))
      : [];

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          p: 4,
          minHeight: "100vh",
          background: darkMode ? "#020617" : "#f1f5f9",
        }}
      >
        {/* HEADER */}

        <Box display="flex" justifyContent="space-between" mb={4}>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              background: "linear-gradient(90deg,#3b82f6,#9333ea)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            🚀 AutoML Intelligence Dashboard
          </Typography>

          <FormControlLabel
            sx={{ color: textColor }}
            control={
              <Switch
                checked={darkMode}
                onChange={() => setDarkMode(!darkMode)}
              />
            }
            label="Dark Mode"
          />
        </Box>

        {/* UPLOAD */}

        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Box
              {...getRootProps()}
              sx={{
                border: "2px dashed #3b82f6",
                p: 4,
                textAlign: "center",
                borderRadius: 2,
                cursor: "pointer",
              }}
            >
              <input {...getInputProps()} />

              {file ? (
                <Typography sx={{ color: textColor }}>
                  Selected File: {file.name}
                </Typography>
              ) : (
                <Typography sx={{ color: textColor }}>
                  Drag CSV Dataset Here
                </Typography>
              )}
            </Box>

            <Button sx={{ mt: 3 }} variant="contained" onClick={trainModel}>
              Train Model
            </Button>

            {loading && <CircularProgress sx={{ ml: 2 }} />}
          </CardContent>
        </Card>

        {error && <Alert severity="error">{error}</Alert>}

        {data && (
          <Box>
            {/* BEST MODEL */}

            <Grid container spacing={3} mb={4}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5">Best Model</Typography>

                    <Chip
                      label={data.best_model}
                      color="primary"
                      sx={{ mt: 1 }}
                    />

                    <Typography sx={{ mt: 2 }}>
                      Accuracy: {(data.accuracy * 100).toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5">Bias Report</Typography>

                    <Typography>
                      Gender Bias: {data.bias_report.gender_bias}
                    </Typography>

                    <Typography>
                      City Bias: {data.bias_report.city_bias}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* MODEL COMPARISON */}

            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h5">Model Accuracy Comparison</Typography>

                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={modelScores}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="score" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* METRICS */}

            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h5">
                  Classification Performance Metrics
                </Typography>

                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={metricsChart}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="value" fill="#22c55e" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* FEATURE IMPORTANCE */}

            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h5">Top Influential Features</Typography>

                <ResponsiveContainer width="100%" height={400}>
                  <BarChart layout="vertical" data={featureImportance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="value" fill="#9333ea" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* ROC CURVE */}

            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h5">ROC Curve</Typography>

                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={rocData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="fpr" />
                    <YAxis />
                    <Tooltip contentStyle={tooltipStyle} />

                    <Area
                      type="monotone"
                      dataKey="tpr"
                      stroke="none"
                      fill="#ef4444"
                      fillOpacity={0.2}
                    />

                    <Line
                      type="monotone"
                      dataKey="tpr"
                      stroke="#ef4444"
                      strokeWidth={3}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* CONFUSION MATRIX + PREDICTION */}

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" sx={{ mb: 3 }}>
                      Confusion Matrix
                    </Typography>

                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 3,
                      }}
                    >
                      {[
                        {
                          label: "True Negative",
                          value: data.confusion_matrix[0][0],
                          color: "#bfdbfe",
                        },
                        {
                          label: "False Positive",
                          value: data.confusion_matrix[0][1],
                          color: "#fecaca",
                        },
                        {
                          label: "False Negative",
                          value: data.confusion_matrix[1][0],
                          color: "#fde68a",
                        },
                        {
                          label: "True Positive",
                          value: data.confusion_matrix[1][1],
                          color: "#bbf7d0",
                        },
                      ].map((cell) => (
                        <Box
                          key={cell.label}
                          sx={{
                            borderRadius: 3,
                            background: cell.color,
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "center",
                            alignItems: "center",
                            p: 3,
                          }}
                        >
                          <Typography sx={{ color: "#000", fontWeight: 600 }}>
                            {cell.label}
                          </Typography>

                          <Typography
                            sx={{
                              fontSize: 42,
                              fontWeight: 800,
                              color: "#000",
                            }}
                          >
                            {cell.value}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* PREDICTION */}

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5">Predict Conversion</Typography>

                    <TextField id="age" label="Age" fullWidth sx={{ mt: 2 }} />
                    <TextField
                      id="income"
                      label="Income"
                      fullWidth
                      sx={{ mt: 2 }}
                    />
                    <TextField
                      id="city"
                      label="City"
                      fullWidth
                      sx={{ mt: 2 }}
                    />

                    <FormControl fullWidth sx={{ mt: 2 }}>
                      <InputLabel>Gender</InputLabel>

                      <Select id="gender">
                        <MenuItem value="Male">Male</MenuItem>
                        <MenuItem value="Female">Female</MenuItem>
                      </Select>
                    </FormControl>

                    <TextField
                      id="visits"
                      label="Website Visits"
                      fullWidth
                      sx={{ mt: 2 }}
                    />
                    <TextField
                      id="time"
                      label="Time Spent"
                      fullWidth
                      sx={{ mt: 2 }}
                    />

                    <Button
                      variant="contained"
                      sx={{ mt: 3 }}
                      onClick={predict}
                    >
                      Predict
                    </Button>

                    {prediction !== null && (
                      <Box mt={3} textAlign="center">
                        <Typography variant="h6">
                          Prediction: {prediction}
                        </Typography>

                        {prediction !== null && (
                          <Box
                            mt={3}
                            sx={{
                              p: 3,
                              borderRadius: 2,
                              textAlign: "center",
                              background:
                                prediction === 1
                                  ? "linear-gradient(135deg,#dcfce7,#bbf7d0)"
                                  : "linear-gradient(135deg,#fee2e2,#fecaca)",
                            }}
                          >
                            <Typography
                              variant="h5"
                              fontWeight="bold"
                              sx={{
                                color: prediction === 1 ? "#15803d" : "#b91c1c",
                              }}
                            >
                              {prediction === 1
                                ? "✅ Customer Likely to Convert"
                                : "❌ Customer Unlikely to Convert"}
                            </Typography>

                            {probability && (
                              <>
                                <Typography mt={1} sx={{ fontWeight: 600 }}>
                                  Conversion Probability:{" "}
                                  {(probability * 100).toFixed(2)}%
                                </Typography>

                                <Box
                                  sx={{
                                    mt: 2,
                                    height: 10,
                                    borderRadius: 5,
                                    background: "#e5e7eb",
                                    overflow: "hidden",
                                  }}
                                >
                                  <Box
                                    sx={{
                                      width: `${probability * 100}%`,
                                      height: "100%",
                                      background:
                                        probability > 0.6
                                          ? "#22c55e"
                                          : probability > 0.4
                                            ? "#facc15"
                                            : "#ef4444",
                                    }}
                                  />
                                </Box>
                              </>
                            )}
                          </Box>
                        )}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
}

export default App;
