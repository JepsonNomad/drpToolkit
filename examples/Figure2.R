#### Import packages ----
library(tidyverse)
library(raster)
library(lubridate)
library(cowplot)
library(rjson)
library(sf)
library(RStoolbox)

#### Import data ----
## > Greenness data ----
alGCC = read_csv("prepped/aligned/extract.csv") %>%
  mutate("Alignment" = "Aligned")
naGCC = read_csv("prepped/extract.csv") %>%
  mutate("Alignment" = "Not aligned")


## > Raster data ----
# Dates
myRdates = c("2018-08-15",
             "2018-11-15",
             "2019-05-15",
             "2019-08-15")
# With alignment
alR1 = raster::stack("prepped/aligned/GB-03_2018_08_14_120000.JPG")
alR2 = raster::stack("prepped/aligned/GB-03_2018_11_14_120001.JPG")
alR3 = raster::stack("prepped/aligned/GB-03_2019_05_14_120000.JPG")
# Unaligned
naR1 = raster::stack("prepped/GB-03_2018_08_14_120000.JPG")
naR2 = raster::stack("prepped/GB-03_2018_11_14_120001.JPG")
naR3 = raster::stack("prepped/GB-03_2019_05_14_120000.JPG")


## > ROI polygons ----
ROI_csv = read_csv("~/Documents/Post_Lab/Data/SNBS/PhenoCams_Polygons/Spp_Polygons_CJ/GB-03/GB-03.csv")
head(ROI_csv)



#### Data wrangling ----
## > GCC timeseries ----
GCC = rbind(alGCC, naGCC)



## > Raster fortification ----
# Convert to lists for iteration
alRList = list(alR1,
               alR2,
               alR3)
naRList = list(naR1,
               naR2,
               naR3)
alRflist = lapply(alRList,
                  function(x){
                    y = fortify(x,
                                maxpixels = 5e6)
                    names(y) <- c("lon","lat","R","G","B")
                    return(y)})
naRflist = lapply(naRList,
                  function(x){
                    y = fortify(x,
                                maxpixels = 5e6)
                    names(y) <- c("lon","lat","R","G","B")
                    return(y)})

for(i in 1:length(alRflist)){
  alRflist[i][[1]]$date = rep(myRdates[i],
                         nrow(alRflist[i][[1]]))
  alRflist[i][[1]]$Alignment = rep("Aligned",
                              nrow(alRflist[i][[1]]))
  naRflist[i][[1]]$date = rep(myRdates[i],
                         nrow(naRflist[i][[1]]))
  naRflist[i][[1]]$Alignment = rep("Not aligned",
                              nrow(naRflist[i][[1]]))
}

alRdf = do.call(rbind,
                alRflist)
naRdf = do.call(rbind,
                naRflist)
# Complete dataset
rasDF = rbind(alRdf,
              naRdf)


## > ROI conversion to sf ----
# Get ROI metadata
roi_id = ROI_csv$region_id
roi_species = lapply(X = ROI_csv$region_attributes,
                     FUN = fromJSON) %>%
  unlist()
roi_info = data.frame("id" = roi_id,
                      "species" = roi_species)
# Extract json info
roi_vertices = lapply(X = ROI_csv$region_shape_attributes,
                      FUN = function(x){
                        y = fromJSON(x)
                        z = data.frame(lon = y$all_points_x,
                                       lat = 1108 - y$all_points_y)
                        z = rbind(z, z[1,])
                        return(z)})
# Generate polygons from json info
roi_pol = lapply(X = roi_vertices,
                 FUN = function(x){
                   y = Polygon(x)
                   return(y)
                 })
roi_pols_list = list()
for(i in 1:length(roi_pol)){
  roi_pols_list[i] = Polygons(roi_pol[i],
                              as(roi_id[i],
                                 "character"))
}
# Convert to spatial object
roi_sp = SpatialPolygons(roi_pols_list)
roi_spdf = SpatialPolygonsDataFrame(roi_sp,
                                    data = roi_info,
                                    match.ID = FALSE)
# Check
plot(roi_spdf,
     col = roi_spdf$species)
## Convert to simple features
roi_sf = st_as_sf(roi_spdf)




#### Plot results ----
## > GCC time series ----
fullTS_nLeg = GCC %>%
  mutate(regionFac = as.factor(regionID)) %>%
  mutate(date = ymd_hms(date),
         yrday = yday(date),
         hour = hour(date)) %>%
  filter(hour == 12) %>%
  filter(NDSI > -0.5) %>%
  ggplot(aes(x = as.Date(date),
             y = GCC,
             group = regionFac,
             col = species)) +
  facet_wrap(~Alignment) +
  geom_rect(aes(xmin = min(as.Date(date)),
                xmax = as.Date("2018-11-21"),
                ymin = min(GCC),
                ymax = max(GCC)),
            fill = "grey80",
            col = "grey40") +
  geom_line() +
  geom_point() +
  xlab("Date") +
  ylab("Greenness") +
  scale_color_manual("",
                     values = CJsBasics::KellyCols[2:20]) +
  scale_x_date(date_breaks = "6 months",date_labels = "%Y-%m") +
  CJsBasics::BasicTheme +
  theme(plot.margin = margin(0,0.5,0,0,"cm")) + 
  theme(legend.position = "none")
fullTS_nLeg

subTS_wLeg = GCC %>%
  mutate(regionFac = as.factor(regionID)) %>%
  mutate(date = ymd_hms(date),
         yrday = yday(date),
         hour = hour(date)) %>%
  filter(hour == 12) %>%
  filter(date < "2018-11-21") %>%
  filter(NDSI > -0.5) %>%
  ggplot(aes(x = date,
             y = GCC,
             group = regionFac,
             col = species)) +
  facet_wrap(~Alignment) +
  geom_line() +
  geom_point() +
  annotate(geom = "text",
           x = as.POSIXct(as.Date("2018-11-12")),
           y = 0.39,
           label = "2018") +
  scale_y_continuous(breaks = c(0.30,0.33,0.36, 0.39),
                     limits = c(0.3,
                                0.4)) +
  scale_color_manual("Species",
                     values = CJsBasics::KellyCols[2:20]) +
  ylab("Greenness") +
  CJsBasics::BasicTheme +
  theme(axis.title.x = element_blank())


## > Raster panels ----
rasPlot = rasDF %>%
  mutate(R = R/255,
         G = G/255,
         B = B/255,
         Date = date) %>%
  ggplot() + 
  geom_tile(aes(x=lon, y=lat, fill=rgb(R,G,B))) +
  scale_fill_identity() +
  geom_sf(data = roi_sf,
          aes(col = species),
          fill = "transparent",
          size = 1) +
  scale_color_manual(values = CJsBasics::KellyCols[2:20]) +
  facet_grid(Alignment ~ Date) + 
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        panel.background = element_blank(),
        panel.grid = element_blank(),
        plot.margin = margin(0,0.25,0,0.75,"cm"),
        legend.position = "none")


## > Combined plots ----
TSplot = plot_grid(fullTS_nLeg,
                   subTS_wLeg,
                   ncol = 1,
                   align = "hv",
                   axis = "tbl",
                   labels = c("b","c"),
                   rel_heights = c(0.5,0.5))

Figure2 = plot_grid(rasPlot,
                    TSplot,
                    ncol = 1,
                    labels = c("a",""),
                    rel_heights = c(0.4,
                                    0.6))

ggsave(filename = "Figure2.jpg",
       plot = Figure2,
       width = 16.6,
       height = 16.6,
       units = "cm",
       dpi = 600)
