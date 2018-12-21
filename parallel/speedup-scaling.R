
options(stringsAsFactors = FALSE)

library(magrittr)
library(tidyverse)

det <- read_csv(file.path(root, "parallel/ref-deterministic-timings.csv"))
sto <- read_csv(file.path(root, "parallel/ref-stochastic-timings.csv"))

det %<>% mutate(speedup = det$time[1] / time, cores = as.integer(cores))
sto %<>% mutate(speedup = sto$time[1] / time, cores = as.integer(cores))

# Scaling, ignoring 'bad points'
# alpha ~ -0.67 +- 0.07
det %>% 
  filter(!cores %in% c(1, 72)) %>% 
  lm(log(time) ~ log(cores), data = .) %>% 
  summary()

# best points
# alpha ~ -0.894 +- 0.007
det %>% 
  filter(cores %in% c(2, 4, 8)) %>% 
  lm(log(time) ~ log(cores), data = .) %>% 
  summary()

# alpha ~ -0.981 +- 0.007
sto %>% 
  filter(cores != 1) %>% 
  lm(log(time) ~ log(cores), data = .) %>% 
  summary()


# Amdahl: Tn/T1 = p(1/n - 1) + 1 [n = cores, p = parallel share]
# Tn/T1 = 1/speedup
det %>% 
  filter(cores != 72) %>% 
  mutate(s_inv = 1 / speedup, n_inv = 1 / cores) %>% 
  lm(s_inv ~ n_inv, data = .) %>% 
  summary()

sto %>% 
  mutate(s_inv = 1 / speedup, n_inv = 1 / cores) %>% 
  lm(s_inv ~ n_inv, data = .) %>% 
  summary()
