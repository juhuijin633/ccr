
library(dplyr)
library(tidyr)

setwd("/Users/juhuijin/Desktop/ccr/demand-analysis-repro/code")
source("paths.R")
source(file.path(PROJECT_DIR, "config/style.R"))

#---------- Figure Generating Code ---------

plot_spectral <- function(df, type = "image", p = "none", truncate = 600, 
                          xlab = "Order", ylab = "Singular Values"){
  
  if (type == "image"){ color <- OI_COLORS[1] ; name = "Image" } else { color <- OI_COLORS[2] ; name = "Text" } 
  p_dict = c(none = "Raw", demean = "Demeaned", demean_scale = "Demeaned and Scaled")
  p_name = p_dict[[p]]
  
  df_ = df %>% 
    filter(type == .env$type , post_processing == .env$p) %>% 
    arrange(desc(value)) %>% 
    mutate(order = row_number()) %>% 
    filter(order <= truncate)
  
  ggplot(data = df_) + 
    geom_point(aes(x = order, y = value), color = color) + 
    geom_line(aes(x = order, y = value), color = color) + 
    labs(
      x = xlab, 
      y = ylab, 
      title = paste0("Top ", truncate, " ", ylab, " (", name, " Embedding, ", p_name, ")")
    ) + 
    theme_oi() 
}

# SVD on Text and Image 
# Plotting L2 norm of each embedding

df = read.csv(file.path(PROJECT_DIR, "/data/spectral/singular_values.csv"))

for (p_name in c("none", "demean", "demean_scale")){
  
  for (type_name in c("text", "image")){
    
    plot = plot_spectral(df, type = type_name, p = p_name, truncate = 30, xlab = "Order", 
                         ylab = "Singular Values")
    
    ggsave(plot = plot, filename = file.path(PROJECT_DIR, "figures", paste0(type_name, "_", p_name, "_", "sval.png")), width = 9, height = 6)
  }
}

