library(shiny)
library(ggplot2)
library(reshape2)
library(reticulate)
library(gridExtra)
library(dplyr)

if (!virtualenv_exists("./venv")) {
  virtualenv_create("./venv")
  virtualenv_install("./venv", packages = c("numpy", "pandas"))
}

py_run_string("
import sys
import types
import pickle
import numpy as np
import datetime

class NetworkData:
    def __init__(self, time=None, jaccard_index=None, A=None, Theta=None, l1_penalty=None):
        self.time = time
        self.jaccard_index = jaccard_index
        self.A = A
        self.Theta = Theta
        self.l1_penalty = l1_penalty

gl_module = types.ModuleType('GL')
gl_module.NetworkData = NetworkData
sys.modules['GL'] = gl_module

def load_network_data(file_path):
    with open(file_path, 'rb') as f:
        raw = pickle.load(f)
    names = raw['name']
    records = raw['data_array']
    
    result = []
    for rec in records:
        A_list = rec.A.tolist() if hasattr(rec.A, 'tolist') else rec.A
        Theta_list = rec.Theta.tolist() if hasattr(rec.Theta, 'tolist') else rec.Theta
        result.append({
            'time': rec.time.isoformat(),
            'jaccard_index': float(rec.jaccard_index),
            'l1_penalty': float(rec.l1_penalty),
            'A': A_list,
            'Theta': Theta_list
        })
    return {'names': names, 'data': result}
")

read_network_pkl <- function(filepath) {
  py_result <- py$load_network_data(filepath)
  names_r <- py_result$names
  n <- length(names_r)
  
  ts_df <- data.frame(
    date = as.Date(sapply(py_result$data, function(x) x$time)),
    jaccard_index = sapply(py_result$data, function(x) x$jaccard_index),
    l1_penalty = sapply(py_result$data, function(x) x$l1_penalty)
  )
  
  matrices <- lapply(py_result$data, function(rec) {
    list(
      A = matrix(unlist(rec$A), nrow = n, byrow = TRUE),
      Theta = matrix(unlist(rec$Theta), nrow = n, byrow = TRUE)
    )
  })
  
  list(
    time_series = ts_df,
    matrices = matrices,
    names = names_r
  )
}

pkl_files <- list.files(path = "data", pattern = "\\.pkl$", full.names = TRUE)
if (length(pkl_files) == 0) {
  dir.create("data", showWarnings = FALSE)
  py_run_string("
import numpy as np, datetime, random, string
from GL import NetworkData

def generate_simulated_data(filename='data/demo.pkl', N=12, months=60):
    start_date = datetime.date(2020, 1, 1)
    times = [start_date + pd.DateOffset(months=i) for i in range(months)]
    times = [t.date() for t in times]
    names = [''.join(random.choices(string.ascii_uppercase, k=random.randint(2,4))) for _ in range(N)]
    data_array = []
    cur_j = 0.5
    cur_l = 0.5
    for t in times:
        cur_j = np.clip(cur_j + np.random.normal(0, 0.05), 0, 1)
        cur_l = np.clip(cur_l + np.random.normal(0, 0.02), 0.001, 2.0)
        A = np.random.choice([0,1], size=(N,N), p=[0.7,0.3])
        np.fill_diagonal(A, 0)
        X = np.random.randn(N, N)
        Theta = (X + X.T)/2
        data_array.append(NetworkData(time=t, jaccard_index=cur_j, A=A, Theta=Theta, l1_penalty=cur_l))
    with open(filename, 'wb') as f:
        pickle.dump({'name': names, 'data_array': data_array}, f)
generate_simulated_data()
  ")
  pkl_files <- list.files(path = "data", pattern = "\\.pkl$", full.names = TRUE)
}

ui <- fluidPage(
  titlePanel("GL 图序列演化可视化"),
  sidebarLayout(
    sidebarPanel(
      selectInput("selected_file", "选择数据文件：",
                  choices = pkl_files,
                  selected = pkl_files[1]),
      hr(),
      helpText("下方滑块可调整观察的时间点"),
      sliderInput("time_index", "时间进度：",
                  min = 0, max = 100, value = 0, step = 1),
      hr(),
      textOutput("time_label")
    ),
    mainPanel(
      plotOutput("series_plot", height = "200px"),
      fluidRow(
        column(6, plotOutput("adj_plot")),
        column(6, plotOutput("theta_plot"))
      ),
      hr(),
      plotOutput("network_plot", height = "550px")   # 新增网络图
    )
  )
)


server <- function(input, output, session) {
  
  currentData <- reactive({
    req(input$selected_file)
    read_network_pkl(input$selected_file)
  })
  
  observe({
    dat <- currentData()
    n_times <- nrow(dat$time_series)
    freezeReactiveValue(input, "time_index")
    updateSliderInput(session, "time_index",
                      max = n_times - 1,
                      value = 0)   # 文件切换后直接从 0 开始
  }) %>% bindEvent(currentData())    
  
  idx <- reactive({
    req(currentData())
    as.integer(input$time_index) + 1
  })
  
  output$time_label <- renderText({
    dat <- currentData()
    t <- dat$time_series$date[idx()]
    paste("当前时间：", t)
  })
  
  output$series_plot <- renderPlot({
    dat <- currentData()
    ts <- dat$time_series
    curr_idx <- idx()
    curr_date <- ts$date[curr_idx]
    
    par(mar = c(3, 4, 2, 4))
    plot(ts$date, ts$jaccard_index, type = "l", col = "#2b8cbe", lwd = 1.5,
         ylim = c(-0.05, 1.05), xlab = "", ylab = "Jaccard Index",
         main = "Jaccard Index & L1 Penalty 演化序列")
    par(new = TRUE)
    plot(ts$date, ts$l1_penalty, type = "l", col = "#d95f02", lwd = 1.5,
         axes = FALSE, xlab = "", ylab = "")
    axis(side = 4, col = "#d95f02", col.axis = "#d95f02")
    mtext("L1 Penalty", side = 4, line = 2.5, col = "#d95f02")
    abline(v = curr_date, col = "#de2d26", lty = 2, lwd = 2)
    legend("top", legend = c("Jaccard Index", "L1 Penalty"),
           col = c("#2b8cbe", "#d95f02"), lty = 1, bty = "n")
  })
  
  output$adj_plot <- renderPlot({
    dat <- currentData()
    mats <- dat$matrices[[idx()]]
    A <- mats$A
    Theta <- mats$Theta
    N <- nrow(A)
    names <- dat$names
    
    df <- expand.grid(Row = 1:N, Col = 1:N)
    df$A <- as.vector(t(A))
    df$Theta <- as.vector(t(Theta))
    df$fill <- "white"
    df$fill[df$A == 1 & df$Theta < 0] <- "red"
    df$fill[df$A == 1 & df$Theta > 0] <- "blue"
    
    ggplot(df, aes(x = Col, y = Row, fill = fill)) +
      geom_tile(color = "lightgray", linewidth = 0.3) +
      scale_fill_identity() +
      scale_y_continuous(breaks = 1:N, labels = names, trans = "reverse") +
      scale_x_continuous(breaks = 1:N, labels = names, position = "top") +
      labs(title = paste0("Adjacency Matrix A (", dat$time_series$date[idx()], ")"),
           x = NULL, y = NULL) +
      coord_fixed() +
      theme_minimal() +
      theme(
        plot.title = element_text(margin = margin(b = 25), face = "bold"),
        axis.text.x.top = element_text(angle = 45, hjust = 0, size = 9, margin = margin(t = 5)),
        axis.text.y = element_text(size = 9),
        legend.position = "none"
      )
  })
  
  output$theta_plot <- renderPlot({
    dat <- currentData()
    mats <- dat$matrices[[idx()]]
    Theta <- mats$Theta
    N <- nrow(Theta)
    names <- dat$names
    
    all_theta <- unlist(lapply(dat$matrices, function(m) as.vector(m$Theta)))
    max_abs <- max(abs(all_theta))
    
    melted <- reshape2::melt(Theta, varnames = c("Row", "Col"), value.name = "value")
    
    ggplot(melted, aes(x = Col, y = Row, fill = value)) +
      geom_tile(color = "lightgray", linewidth = 0.3) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                           midpoint = 0, limits = c(-max_abs, max_abs),
                           name = "Theta") +
      scale_y_continuous(breaks = 1:N, labels = names, trans = "reverse") +
      scale_x_continuous(breaks = 1:N, labels = names, position = "top") +
      labs(title = paste0("Precision Matrix Theta (", dat$time_series$date[idx()], ")"),
           x = NULL, y = NULL) +
      coord_fixed() +
      theme_minimal() +
      theme(
        plot.title = element_text(margin = margin(b = 25), face = "bold"),
        axis.text.x.top = element_text(angle = 45, hjust = 0, size = 9, margin = margin(t = 5)),
        axis.text.y = element_text(size = 9)
      )
  })
  
  output$network_plot <- renderPlot({
    dat <- currentData()
    mats <- dat$matrices[[idx()]]
    A <- mats$A
    Theta <- mats$Theta
    N <- nrow(A)
    names <- dat$names
    
    angles <- seq(0, 2 * pi, length.out = N + 1)[1:N]
    x_nodes <- cos(angles)
    y_nodes <- sin(angles)
    node_df <- data.frame(x = x_nodes, y = y_nodes, name = names)
    
    edges_list <- list()
    for (i in 1:(N - 1)) {
      for (j in (i + 1):N) {
        if (A[i, j] == 1) {   # 存在边
          sign_theta <- sign(Theta[i, j])
          edge_type <- if (sign_theta > 0) {
            "positive"
          } else if (sign_theta < 0) {
            "negative"
          } else {
            "zero"
          }
          edges_list[[length(edges_list) + 1]] <- data.frame(
            x = x_nodes[i], y = y_nodes[i],
            xend = x_nodes[j], yend = y_nodes[j],
            type = edge_type
          )
        }
      }
    }
    edges_df <- do.call(rbind, edges_list)
    
    color_map <- c("positive" = "#2166ac",   # 深蓝
                   "negative" = "#b2182b",   # 深红
                   "zero"     = "gray60")    # 灰色
    
    ggplot() +
      geom_segment(data = edges_df, 
                   aes(x = x, y = y, xend = xend, yend = yend, color = type),
                   linewidth = 1.2) +
      geom_point(data = node_df, aes(x, y), size = 8, shape = 21, 
                 fill = "white", color = "black", stroke = 1.2) +
      geom_text(data = node_df, aes(x, y, label = name), size = 4, fontface = "bold") +
      scale_color_manual(values = color_map,
                         name = "Edge sign",
                         breaks = c("positive", "negative", "zero"),
                         labels = c("Positive (θ>0)", "Negative (θ<0)", "Zero (θ=0)")) +
      coord_fixed(clip = "off") +
      labs(title = paste0("Network (", dat$time_series$date[idx()], ")")) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        legend.position = "bottom",
        legend.title = element_text(face = "bold")
      ) +
      xlim(-1.3, 1.3) + ylim(-1.3, 1.3)
  })
}

shinyApp(ui = ui, server = server)
