suppressPackageStartupMessages({library(arrow)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)
library(RColorBrewer)
library(patchwork)
library(IRdisplay)})


# setting signature stats path
signatures_stats_path <- file.path("../results/signatures/signature_importance.csv")
if (!file.exists(signatures_stats_path)) {
  stop(paste("File not found:", signatures_stats_path))
}

# setting output path for the generated plot
sig_plot_output_dir = file.path("./figures")
if (!file.exists(sig_plot_output_dir)) {
  stop(paste("File not found:", signatures_stats_path))
}

# load feature space config signatures_stats
sig_stats_df <- read.csv(signatures_stats_path)
head(sig_stats_df)


# Render figure size larger
height <- 8
width <- 16  # Increased for two plots side by side
options(repr.plot.width = width, repr.plot.height = height)

# Extract channel from feature names
sig_stats_df$channel <- sapply(strsplit(sig_stats_df$feature, "_"), `[`, 1)

# Set up color palettes
n_channels <- length(unique(sig_stats_df$channel))
dark2_palette <- brewer.pal(min(n_channels, 8), "Dark2")

# Define y-axis limits (extend beyond max to give more space, excluding infinite values)
y_max <- max(sig_stats_df$neg_log10_p_value[is.finite(sig_stats_df$neg_log10_p_value)], na.rm = TRUE) * 1.1

# Plot 1: Colored by Channel
plot_channel <- ggplot(sig_stats_df, aes(x = ks_stat, y = neg_log10_p_value, color = channel)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = dark2_palette) +
  scale_y_continuous(limits = c(0, y_max), expand = expansion(mult = c(0.02, 0.05))) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40", linewidth = 0.5) +
  labs(
    x = "KS statistic (effect size)",
    y = "-log10(FDR-corrected p-value)",
    title = "Feature significance by channel",
    color = "Channel"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )

# Plot 2: Colored by Significant (on/off)
plot_significant <- ggplot(sig_stats_df, aes(x = ks_stat, y = neg_log10_p_value, color = signature)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(
    values = c("off" = "gray60", "on" = "#E41A1C"),
  ) +
  scale_y_continuous(limits = c(0, y_max), expand = expansion(mult = c(0.02, 0.05))) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40", linewidth = 0.5) +
  labs(
    x = "KS statistic (effect size)",
    y = "-log10(FDR-corrected p-value)",
    title = "Feature significance by signature",
    color = "Signature"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )

# Combine plots side by side
combined_plot <- plot_channel + plot_significant

# Save plot
ggsave(file.path(sig_plot_output_dir, "cfret_signature_significance_plots.png"), combined_plot, width = 16, height = 8, dpi = 300)

# Display
combined_plot
