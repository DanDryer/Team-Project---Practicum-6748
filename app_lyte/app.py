import streamlit as st


def about():
    st.markdown("<h1 style='text-align: center; color: black;'>Atmospheric Carbon Dioxide and Socioeconomic Modeling Via Cluster Analysis</h1>",
                unsafe_allow_html=True)
    st.markdown(
        """<a style='display: block; text-align: center;' href="https://github.com/DanDryer/Team-Project-Practicum-6748">View Code Repository</a>""", unsafe_allow_html=True,)

    st.divider()
    st.markdown("""<p style='text-align: left; color: black; font-size: 20px;'>Natural gas and coal accounted for roughly 58% of the United States electricity generation in 2022, releasing approximately 1,508 million metric tons of Carbon Dioxide (CO<sub>2</sub>) <sup>1</sup>. As atmospheric CO<sub>2</sub> levels
                increase, it is likely that temperatures will continue to rise due to the atmospheric greenhouse effect <sup>2</sup>. There is a need to reduce our CO<sub>2</sub> emissions accross the globe. 
                In the United States, identifying CO<sub>2</sub> sources and sinks and potential driving factors is integral to tackling the problem. Specifically, can we group US counties based on socioeconomic factors and atmospheric CO<sub>2</sub> levels?
                If successful, this data can be utilized to better drive policy creation for specific geographic regions thereby mitigating CO<sub>2</sub> emissions.</p>""",
                unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: grey;'>Atmospheric Carbon Dioxide Levels Contiue to Rise in The United States</h2>", unsafe_allow_html=True)
    htmlFile1 = open("app_lyte/images/co2_vs_time.html",
                     'r', encoding='utf-8')
    fig1 = htmlFile1.read()
    st.components.v1.html(fig1, height=430)

    st.markdown(
        " ### Atmospheric Carbon Dioxide Data - Collected by Orbiting Satellites Using Advanced Remote Sensing Techniques")
    st.markdown("""<p style='text-align: left; color: black; font-size: 20px;'> Atmospheric CO<sub>2</sub> data was obtained from NASA's Orbiting Carbon Observatory-2 (OCO-2) and Japan's Aerospace Exploration Agency's Greenhouse Gases Orbiting Satellite (GOSAT). Merging these datasets and using further geostatistical techniques,
                researchers Sheng et al. produced a more complete CO<sub>2</sub> emissions dataset. The open source dataset was utilized for this project <sup>3</sup> ; it contains 357153 unique data points, each detailing 
                a column averaged CO<sub>2</sub> reading, the date of the reading, and the 
                geospatial coordinates. This data underwent further cleaning, filtering, and grouping.</p>""",
                unsafe_allow_html=True)

    st.markdown(
        " ### Socioeconomic Data - Collected from the United States Census and the Centers of Disease Control")
    st.markdown("""<p style='text-align: left; color: black; font-size: 20px;'> The United States Census collects data pertaining to socioeconomic factors. The Centers for Disease Control calculate and publish a Social Vulnerability Index (SVI) every two years in tabular format with numerous geographic
                identifiers <sup>4</sup>. SVI meaures social conditions of a community and its vulnerability to human suffering and financial loss in the event of a disastor. Some of the measurements that were aggregated include: persons below 150% poverty, persons unemployed, 
                aged 65 & older, civilians with a disability, etc. This data set contains 82269 unique data points with 109 fields. This data underwent further merging, cleaning, filtering, and grouping.</p>""",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("<h2 style='text-align: center; color: grey;'>Can we Connect Atmospheric CO<sub>2</sub> and Socioeconomic Vulnerability Levels in the US? </h2>",
                unsafe_allow_html=True)

    left, right = st.columns((.7, 2))
    with left:
        mapbox = st.radio(
            "Select Map To Explore",
            ["Atmospheric CO2", "Socioeconomic Factors"],
            captions=["Navigate atmospheric CO<sub>2</sub> hot zones", "Interact with Socioeconomic Vulnerability Indicators"])
    with right:
        if mapbox == 'Atmospheric CO2':
            st.markdown("<h3 style='text-align: left ; color: black;'>Average Annual Atmospheric CO<sub>2</sub> (ppm) by County</h2>",
                        unsafe_allow_html=True)
            htmlFile2 = open("app_lyte/images/countyco2_vs_year.html",
                             'r', encoding='utf-8')
            fig2 = htmlFile2.read()
            st.components.v1.html(fig2, height=600)
        else:
            with left:
                option = st.selectbox(
                    'Select an Indicator',
                    ('Overall Vulnerability', 'Socioeconoimc Status', 'Household Characteristics', 'Racial & Ethnic Minority Status', 'Housing Type & Transportation'))

            if option == 'Overall Vulnerability':
                st.markdown("<h3 style='text-align: left; color: black;'>Overall Summary Ranking by County</h3>",
                            unsafe_allow_html=True)
                htmlFile3 = open("app_lyte/images/svi_vs_year.html",
                                 'r', encoding='utf-8')
                fig3 = htmlFile3.read()
                st.components.v1.html(fig3, height=600)

            elif option == 'Socioeconoimc Status':
                st.markdown("<h3 style='text-align: left; color: black;'>Socioeconomic Status by County</h3>",
                            unsafe_allow_html=True)
                htmlFile3 = open("app_lyte/images/soc_econ_status.html",
                                 'r', encoding='utf-8')
                fig3 = htmlFile3.read()
                st.components.v1.html(fig3, height=600)

            elif option == 'Household Characteristics':
                st.markdown("<h3 style='text-align: left; color: black;'>Household Characteristics by County</h3>",
                            unsafe_allow_html=True)
                htmlFile3 = open("app_lyte/images/house_char.html",
                                 'r', encoding='utf-8')
                fig3 = htmlFile3.read()
                st.components.v1.html(fig3, height=600)

            elif option == 'Racial & Ethnic Minority Status':
                st.markdown("<h3 style='text-align: left; color: black;'>Racial & Ethnic Minority Status by County</h3>",
                            unsafe_allow_html=True)
                htmlFile3 = open("app_lyte/images/rac_ethn_min_status.html",
                                 'r', encoding='utf-8')
                fig3 = htmlFile3.read()
                st.components.v1.html(fig3, height=600)

            elif option == 'Housing Type & Transportation':
                st.markdown("<h3 style='text-align: left; color: black;'>Housing Type & Transportation by County</h3>",
                            unsafe_allow_html=True)
                htmlFile3 = open("app_lyte/images/hous_transp.html",
                                 'r', encoding='utf-8')
                fig3 = htmlFile3.read()
                st.components.v1.html(fig3, height=600)
    with left:
        st.markdown("""<p style='text-align: left; color: black;'>Hover over map for additional details</p>""",
                    unsafe_allow_html=True)

    st.divider()

    st.markdown("<h2 style='text-align: center; color: grey;'>Clustering Analysis</h2>",
                unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; color: black; font-size: 20px;'>The aim of clustering analysis is to identify groups of similar objects, where clustered objects are more like one another than those in seperate clusters <sup>5</sup>. 
                Multiple clustering approaches were tested and performance was compared using silhouette scores and the Davies-Bouldin index. The best performing approach was K-means clustering with a silhoutte score
                and Davies-Bouldin index of 0.355 and 1.258, respectively. Based on these results, the clusters are not well-seperated since the sihouette scores are low while the Davies-Bouldin index is high. However, this clustering did lead to
                some interesting results with clusters appearing to seperate relatively well geographically. Cluster characteristics and grouping can be explored in the below interactive radar chart and map.
                This relationship is worth further exploring with cluster analysis using other data sources and computational techniques. 
                </p>""",
                unsafe_allow_html=True)

    left, right = st.columns((2))
    with right:
        cluster_map_html = open(
            "app_lyte/images/clustered.html", 'r', encoding='utf-8')
        cluster_fig = cluster_map_html.read()
        st.components.v1.html(cluster_fig, height=400)
        st.markdown("""<p style='text-align: center; color: black;'>Counties highlighted by cluster assignment. Colors and id match those displayed in the radar plot.</p>""",
                    unsafe_allow_html=True)
    with left:
        radar = open(
            "app_lyte/images/radar.html", 'r', encoding='utf-8')
        radar_fig = radar.read()
        st.components.v1.html(radar_fig, height=400, width=800)
        st.markdown("""<p style='text-align: center; color: black;'>Approximated cluster characteristics by averaged socioeconomic indicators within each cluster.
                    Individual clusters can be selected for viewing in the legend to the right of the radar plot</p>""",
                    unsafe_allow_html=True)

    st.divider()

    st.markdown("<h2 style='text-align: center; color: grey;'>References</h2>",
                unsafe_allow_html=True)

    st.write(
        "1. Electricity in the U.S. - U.S. Energy Information Administration (EIA). (n.d.). Homepage - U.S. Energy Information Administration (EIA). [Link](%s)" % "https://www.eia.gov/energyexplained/electricity/electricity-in-the-us.php")
    st.write(
        """2. IPCC, 2023: Climate Change 2023: Synthesis Report. Contribution of Working Groups I, II and III to the Sixth Assessment Report
        of the Intergovernmental Panel on Climate Change [Core Writing Team, H. Lee and J. Romero (eds.)]. IPCC, Geneva, Switzerland,
        184 pp., doi: 10.59327/IPCC/AR6-9789291691647 [Link](%s)""" % "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_FullVolume.pdf """)

    st.write(
        """3. Sheng, Mengya; Lei ,Liping; Zeng, Zhao-Cheng; Weiqiang Rao; Hao Song; Wu, Changjiang, 2021, "Global land 1° mapping XCO2 dataset using satellite observations of GOSAT and OCO-2 from 2009 to 2020", June 15, 2023., Harvard Dataverse, V4  [Link](%s)""" % "https://doi.org/10.7910/DVN/4WDTD8""")
    st.write(
        """4. Centers for Disease Control and Prevention/ Agency for Toxic Substances and Disease Registry/ Geospatial Research,
        Analysis, and Services Program. CDC/ATSDR Social Vulnerability Index [2020, 2018, 2016, 2014]
        Database US. [Link](%s)""" % "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html """)
    st.write(
        """5. Govender, P., & Sivakumar, V. (2020). Application of k-means and hierarchical clustering techniques for analysis of air pollution: 
        A review (1980-2019). Atmospheric Pollution Research, 11(1), 40-56. [Link](%s)""" % "https://doi.org/10.1016/j.apr.2019.09.009 """)


def main():

    # Page configuration
    st.set_page_config(page_title="CO2-Socioeconomic Modeling",
                       page_icon=":satellite:", layout="wide")

    about()


if __name__ == "__main__":
    main()
