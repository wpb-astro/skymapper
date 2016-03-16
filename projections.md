#Sky projections

I developed an appreciation of cartography over the question:

> Which map projection(s) should we choose to truthfully represent data from the celestial sphere in 2D? 

Note: truthfully, and celestial.

When you look at [widely used projections](http://egsc.usgs.gov/isb//pubs/MapProjections/projections.html), they (can) have four main characteristics (in brackets: the feature they preserve when you move around the map):

1. equidistant (distance)
2. conformal (angle)
3. equal-area (size)
4. perspective (relation between distance and size)

It is unfortunate but inevitable that any mapping from a curved onto a flat surface can only preserve some of those features, but not all, and in some cases none. So which ones matter most?

This is where truthfully comes in. When studying visual perception, it turns out that humans can distinguish some features better than others (see the seminal work by [Cleveland & McGill, 1984](https://web.cs.dal.ca/~sbrooks/csci4166-6406/seminars/readings/Cleveland_GraphicalPerception_Science85.pdf)). The order of the list above reflects that: distances (along a common scale, as in a bar chart) can most precisely be distinguished, followed by objects with different angles (think: trend lines or model predictions in a scatter plot), sizes, and volumes (for which perspective is relevant). Only after that will saturation and color be incorporated to distinguish objects.

So, we should be using equidistant maps, right? No.

The questions we face is not how precisely or rapidly can someone perceive features of the map. It's how well can we show what we want to show.

Let's take an equidistant map. If I found myself on an airplane with a finite amount of fuel, that's exactly the map I'd want to have available to me. In astronomical terms, it'll tell us how long it takes to slew a telescope, but most viewers won't care about that.

Conformal maps face a similar problem. Outside of navigation, it's rare that angles are of great importance over large areas—and on small areas most of what we discuss here is irrelevant. One can alter the angles of the stellar constellations quite a bit, and they will remain recognizable. For interpreting e.g. the effects of the moon illumination, a conformal map might be useful, but again: not necessarily the main focus here.

Equal-area maps are interesting because size matters, specifically for wide-field survey maps. The reason is that we often want to show particularly *large* features. Think of large cosmic voids, which we may want to compare visually to the size of the full moon. It would be odd if the size of those features change if we move them around on the map. Hence, I argue that **most wide-field maps should be equal-area** (or at least close to).

Brings us to the perspective projections. They feel natural because they mimic how we would look *upon* a sphere from some distance. Remember: celestial. We view that sphere from the inside! So, while we should clearly show the curved nature of the sky, the classical perspective is not helping here.

These considerations leave only a few projections for hemisphere-sized maps

1. Lambert Azimuthal Equal-Area
2. Albers Equal-Area Conic
3. [Lambert Conformal Conic, which is not exactly equal area].

and another few for all-sky maps:

1. Lambert Cylindrical Equal-Area and variants (like Behrmann)
2. Mollweide
3. Eckert IV
4. Healpix and others

For the whole sky, we essentially have to throw out the requirement of natural perspective altogether. So let's focus on those that work on a smaller area.

##(Close to) Equal-area projections for ~20,000 sq. degrees 

The azimuthal projection is great for an area that extends equally in North-South and in East-West direction, while both of the conic projections work best when the extension of the data is mainly East-West (because they have to squash North-South distances to maintain equal-area). For surveys observed from one fixed location on the ground, the latter is the more typical situation because the range of visible declinations is limited, whereas RA is not. The conics have this property for all reasonably moderate latitudes, while the cylindrical ones work best only along the equator (and have the sphere appear without any curvature at all, which works against any perspective interpretation).

Leaves **Albers Equal-Area** and **Lambert Conformal** conics. Truth be told, they are pretty close in most regards (which is why both are implemented in *skymapper*). Albers is strictly equal area at the expense of rather strong distortions towards the poles. Lambert preserves sizes only along the standard parallels, but is otherwise very well balanced. In particular, it shows the pole as a point, unlike the Albers, where it's an arc (see [e.g. here](http://lazarus.elte.hu/~guszlev/vet/conic.htm)).

In essence, if you stay in moderate latitudes the equal-area character of the Albers projection is preferable. If you need to approach or cover the poles, Lambert is a better choice.