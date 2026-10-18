
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

df = read.csv(file.path(PROJECT_DIR, "/data/spectral/singular_values.csv"))

for (p_name in c("none", "demean", "demean_scale")){
  
  for (type_name in c("text", "image")){
    
    plot = plot_spectral(df, type = type_name, p = p_name, truncate = 30, xlab = "Order", 
                         ylab = "Singular Values")
    
    ggsave(plot = plot, filename = file.path(PROJECT_DIR, "figures", paste0(type_name, "_", p_name, "_", "sval.pdf")), width = 9, height = 6)
  }
}

p_name = "demean_scale"
type_name = "text"
plot_spectral(df, type = type_name, p = p_name, truncate = 30, xlab = "Order", 
              ylab = "Singular Values")

image_sval_plot = plot_spectral(singular_vals, type = "image", p = "none", 
                                truncate = 30, xlab = "Order", ylab = "Singular Values")

#---- save plots 

ggsave(plot = image, filename = file.path(PROJECT_DIR, "figures/image_spectral_decomp.png"), width = 9, height = 6)
ggsave(plot = image_log, filename = file.path(PROJECT_DIR, "figures/image_spectral_decomp_log.png"), width = 9, height = 6)
ggsave(plot = text, filename = file.path(PROJECT_DIR, "figures/text_spectral_decomp.png"), width = 9, height = 6)
ggsave(plot = text_log, filename = file.path(PROJECT_DIR, "figures/text_spectral_decomp_log.png"), width = 9, height = 6)


#----------------- MEETING 10/16 ------------------# 

norm_L2 = read.csv(file.path(PROJECT_DIR, "data/spectral/embedding_L2_norms.csv"))

text_norm_L2 = ggplot(norm_L2 %>% 
                        filter(type == "text") %>% 
                        arrange(desc(norm_L2)) %>% 
                        mutate(i = row_number())) + 
  geom_point(aes(x = i, y = norm_L2), color = OI_COLORS[1]) + 
  labs(
    x = "Order", 
    y = "L2 Norm", 
    title = "Variance of Text Embeddings"
  ) + 
  theme_oi()

image_norm_L2 = ggplot(norm_L2 %>% 
                        filter(type == "image") %>% 
                        arrange(desc(norm_L2)) %>% 
                        mutate(i = row_number())) + 
  geom_point(aes(x = i, y = norm_L2), color = OI_COLORS[2]) + 
  labs(
    x = "Order", 
    y = "L2 Norm", 
    title = "Variance of Image Embeddings"
  ) + 
  theme_oi()


text_norm_L2
image_norm_L2

#----- Save plots 

ggsave(plot = image_norm_L2, filename = file.path(PROJECT_DIR, "figures/image_norm_L2.png"), width = 9, height = 6)
ggsave(plot = text_norm_L2, filename = file.path(PROJECT_DIR, "figures/text_norm_L2.png"), width = 9, height = 6)




