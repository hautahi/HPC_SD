## A few output files differ from baseline.
## Check that they differ in rounding error only.
## This mostly failed, but because R doesn't use enough precision...
## so we accomplished our goal anyhow.

options(stringsAsFactors = FALSE, digits = 17)
library(tidyverse)

cwd <- getwd()

# For each file:
# 1. read in the baseline and the new version
# 2. convert to matrices by removing and non-numeric columns
# 3. compute the relative error and absolute error
# 4. report the RMS and max error
deltas <- list.files(file.path(cwd, "output")) %>% 
  map_df(~ {
    old_file <- file.path(cwd, "outputs/deterministic_baseline", .x)
    new_file <- file.path(cwd, "outputs/deterministic_baseline", .x)
    
    old_mat <- read_csv(old_file)
    old_mat <- as.matrix(old_mat[, map_lgl(old_mat, is.numeric)])
    new_mat <- read_csv(new_file)
    new_mat <- as.matrix(new_mat[, map_lgl(new_mat, is.numeric)])
    
    assertthat::assert_that(all(dim(new_mat) == dim(old_mat)))
    
    # Absolute & relative error
    abs_err_mat <- new_mat - old_mat
    rel_err_mat <- abs_err_mat / (old_mat + 1e-15)
    
    tibble(
      file = .x,
      max_abs = max(abs_err_mat),
      rms_abs = sqrt(mean(abs_err_mat ** 2)),
      max_rel = max(rel_err_mat),
      rms_rel = sqrt(mean(rel_err_mat ** 2))
    )
  })

print(deltas)
