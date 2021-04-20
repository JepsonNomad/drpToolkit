library(tidyverse)
library(scales)


GCCnoAl = read_csv("GB-03/prepped/extract.csv")
GCCwiAl = read_csv("GB-03/prepped/aligned/extract.csv")



myGCC %>%
  mutate(regionFac = as.factor(regionID)) %>%
  mutate(dateDate = as.Date(date)) %>%
  filter(NDSI > 0) %>%
  ggplot(aes(x = dateDate,
             y = GCC,
             group = regionFac,
             col = species)) +
  geom_point() +
  geom_line(size = 0.1) +
  scale_color_manual("Species",
                     values = CJsBasics::KellyCols[2:20]) +
  scale_x_date(labels = date_format("%b %Y")) +
  xlab("Date") +
  theme_classic() +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 14))

myGCC %>%
  mutate(regionFac = as.factor(regionID)) %>%
  mutate(dateDate = as.Date(date)) %>%
  ggplot(aes(x = dateDate,
             y = NDSI,
             group = regionFac,
             col = species)) +
  geom_point() +
  scale_color_manual("Species",
                     values = CJsBasics::KellyCols[2:20]) +
  scale_x_date(labels = date_format("%b %Y")) +
  xlab("Date") +
  theme_classic() +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 14))
