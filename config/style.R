#' The OI color palette
#' @export
OI_COLORS <- c(
  "#29B6A4", "#FAA523", "#003A4F", "#7F4892", "#A4CE4E",
  "#2B8F43", "#0073A2", "#E54060", "#FFD400", "#6BBD45"
)

#' @export
scale_color_oi <- function(...) scale_color_manual(values = OI_COLORS, ...)

#' @export
scale_fill_oi <- function(...) scale_fill_manual(values = OI_COLORS, ...)


#' ggplot clone of the OI Stata scheme
#'
#' @examples
#' line <- ggplot(line_df, aes(x = year, y = lifeExp)) +
#'   geom_line(colour = "#007f7f", size = 1) +
#'   geom_hline(yintercept = 0, size = 1, colour = "#333333") +
#'   oi_style()
#' @export
theme_oi <- function(high_contrast=TRUE, font="Helvetica") {
  colors <- list(
    titles = ifelse(high_contrast, "black", "#222222"),
    axes = ifelse(high_contrast, "black", "#747577")
  )
  
  ggplot2::theme_classic(base_family = font) %+replace%
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        size = 16,
        face = "bold",
        color = colors$titles, 
        margin = ggplot2::margin(b = 15), 
        hjust = 0.5
      ),
      plot.subtitle = ggplot2::element_text(
        size = 12,
        margin = ggplot2::margin(t = -9, b = 16), 
        hjust = 0.5
      ),
      legend.position = "bottom",
      legend.text = ggplot2::element_text(
        size = 16,
        color = colors$titles
      ),
      axis.title = ggplot2::element_text(
        size = 16,
        color = colors$axes
      ),
      axis.text = ggplot2::element_text(
        size = 16,
        color = colors$axes
      ),
      axis.ticks = ggplot2::element_line(color = colors$axes),
      axis.ticks.length = grid::unit(6, "pt"), 
      axis.line = ggplot2::element_line(color = colors$axes),
      strip.text = ggplot2::element_text(size = 12, hjust = 0),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.margin = ggplot2::margin(0.5, 0.5, 0.5, 0.5, "cm"), 
      axis.title.x = ggplot2::element_text(
        hjust = 0.5, 
        margin = margin(t = 10)
      ), 
      axis.title.y = ggplot2::element_text(
        angle = 90, 
        vjust = 0.5,
        margin = margin(r = 10)
      )
    )
}

expand_axis <- function(ymin = 0, expand = c(0, 0)) {
  list(
    scale_y_continuous(limits = c(ymin, NA), expand = expand),
    scale_x_continuous(expand = expand)
  )
}

#' Use the OI theme and color palette for all plots generated in this session
#'
#' @param ... args to pass to `theme_oi`
#'
#' @examples
#' # place this in your .Rprofile to set the theme globally
#' oiplot::set_oi_theme()
#'
#' @export
set_oi_theme <- function(...) {
  # set theme
  ggplot2::theme_set(theme_oi(...))
  
  # set default colors for when there is NO color/fill mapping
  ggplot2::update_geom_defaults("point", list(colour = OI_COLORS[1]))
  ggplot2::update_geom_defaults("line", list(colour = OI_COLORS[1]))
  ggplot2::update_geom_defaults("smooth", list(colour = OI_COLORS[1]))
  
  # set default colors when there IS a color/fill mapping
  options(
    ggplot2.discrete.colour = OI_COLORS,
    ggplot2.discrete.fill = OI_COLORS
  )
}

# old names for backwards compat

#' @export
oi_style <- function(...) {
  .Deprecated("theme_oi")
  theme_oi(...)
}

#' @export
set_oi_palette <- function(...) {
  .Deprecated("set_oi_theme")
  set_oi_theme(...)
}
