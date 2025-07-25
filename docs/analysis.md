# ANALYSIS

# Uniform distribution
The entropy is
$$
H(X)  ~~~~= \int_a^b \frac{dx}{b-a} log_2 (b-a) \\
= log_2(b-a)
$$
where the units are bits.

# Joint continuous discrete distribution
Let $f_X$ be a continous (possibly multivarate) distribution and $f_Y$ be a discrete (possibly multivariate) distribution for $\cal{Y}. Then, we can construct their joint distribution, $f_{X,Y}$ as follows:
$$
\begin{align}
f_{X,Y} (x,y) & = & \sum_{y \in \cal{Y}} f_X (x | y) f_Y (y)
\end{align}
$$
So, their joint entropy is
$$
\begin{align}
H(X,Y) & = & \sum_{y \in \cal{Y}} \int_{C_X} f_{X,Y} log_2( f_{X,Y}) dx \\
& = &  \sum_{y \in \cal{Y}} \int_{C_X} 
 f_X (x | y) f_Y (y) log_2( f_X (x | y) f_Y (y)) dx \\
 & = &  \sum_{y \in \cal{Y}} f_Y(y) \int_{C_X} 
 f_X (x | y) log_2( f_X (x | y)) dx \\
 & & + \sum_{y \in \cal{Y}} f_Y(y) log_2 (f_Y(y))\\
 & = & \sum_{y \in \cal{Y}} f_Y(y) h(X|Y=y) + H(Y) 
\end{align}
$$
where $C_X$ is the region of integration for $X$.