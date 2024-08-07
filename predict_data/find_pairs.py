# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 1:53
# @Author  : HaiqingSun
# @OriginalFileName: extract
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

def extract_common_elements(list1, list2):
    common_elements = []
    for element in list1:
        if element in list2:
            common_elements.append(element)
    return common_elements



indices_12_1 = [230, 231, 272, 278, 302, 314, 316, 317, 318, 328, 357, 364, 386, 392, 411, 419, 429, 434, 435, 447, 454, 459, 460, 501, 505, 545, 546, 547, 557, 561, 617, 663, 676, 689, 730, 734, 762, 774, 776, 786, 815, 850, 869, 887, 889, 892, 903, 917, 918, 928, 937, 939, 955, 959, 962, 963, 970, 973, 996, 997, 999, 1003, 1004, 1005, 1019, 1022, 1073, 1079, 1091, 1098, 1102, 1104, 1106, 1108, 1118, 1121, 1129, 1131, 1132, 1134, 1138, 1141, 1147, 1199, 1244, 1604, 1605, 1650, 1678, 1686, 1692, 1713, 1731, 1765, 1766, 1773, 1785, 1793, 1794, 1797, 1808, 1821, 2136]
indices_12_7 = [231, 318, 434, 447, 460, 689, 762, 892, 917, 918, 928, 959, 963, 997, 999, 1004, 1005, 1019, 1073, 1098, 1104, 1118, 1121, 1129, 1134, 1604]
indices_13_0 = [231, 689, 762, 928, 963, 1005, 1098, 1134]
indices_12_1_GCNGATFG = [470, 471, 496, 522, 594, 620, 622, 642, 646, 653, 685, 690, 714, 718,
           738, 796, 835, 873, 901, 924, 937, 938, 1109, 1340, 1404, 1405, 1418, 1430,
           1456, 1485, 1487, 1498, 1516, 1541, 1554, 1556, 1558, 1564, 1576, 1580, 1587, 1600,
           1601, 1604, 1608, 1610, 1619, 1624, 1626, 1627, 1628, 1634, 1672, 1682, 1688, 1695,
           1712, 1714, 1715, 1730, 1738, 1740, 1769, 1773, 1776, 1786, 1807, 1831, 1835, 1844,
           1847, 1858, 1862, 1871, 1872, 1892, 2014, 2023, 2086, 2119, 2197, 2242, 2269, 2274,
           3273, 3316, 3323, 3326, 3365, 3424, 3425, 3451, 3452, 3492, 3571, 3598, 3639, 3695,
           3712, 3730]
indices_12_7_GCNGATFG = [470, 471, 873, 1554, 1556, 1558, 1730, 1807, 1858, 3695]
indices_12_9_GCNGATFG = [1554, 1558, 1807, 1858]
indices_13_0_GCNGATFG = [1554, 1858]
indices_12_7_GCNFG = [471, 747, 938, 1997, 2025, 2043, 2148, 2197, 2294]
indices_12_4_GCNFG = [471, 642, 690, 747, 868, 873, 893, 938, 1405, 1827, 1872, 1997, 2025, 2043,
                2138, 2148, 2197, 2202, 2208, 2269, 2274, 2294, 2302, 2806, 3740]
indices_12_4_GCNGATFG = [470, 471, 642, 835, 873, 901, 924, 937, 938, 1109, 1456, 1487, 1554, 1556,
                1558, 1627, 1628, 1672, 1730, 1807, 1847, 1858, 2269, 2274, 3273, 3326,
                3424, 3598, 3695, 3730]

# indices_12_1_GATGCN = "4 174 179 222 317 406 1807 4360 4609 8410 8492 8499 8563 8580 8581 8609 8628 8653 8723 8774 8812 8840 8863 9279 9307 12766 13079 13080 13103 13171 13184 13201 13233 13250 13255 13298 13327 13393 13403 13419 13444 13482 13510 15568 16348 16349 16372 16375 16384 16446 16500 16502 16520 16523 16524 16525 16531 16565 16567 16578 16616 16660 16662 16672 16684 16710 16713 16724 16742 16746 16751 16775 16779 16782 16802 16815 16816 16839 16841 16842 16907 16920 16921 16922 16936 16969 16991 16992 17032 17033 17034 17083 17127 17129 17139 17151 17177 17180 17191 17209 17213 17218 17242 17246 17258 17269 18684 18854 19077 19086 32694 32845 32872 33161 33314 33331 33525 33563 34094 34123 34184 34198 34219 34237 34278 34338 34342 34370 34371 34461 34465 34493 34517 34527 34651 34992 35395 35451 35620 35655 35772 35862 35962 35991 36052 36087 36146 36206 36210 36238 36239 36329 36333 36361 36385 37363 37611 37830 37955 37990 38078 38197 38253 38764 38781 38785 38793 38797 38808 38820 38821 38829 38832 38833 38834 38838 38853 38854 38868 38889 38890 38891 38907 38924 38927 38928 38947 38948 38957 38969 38984 39004 39007 39008 39012 39040 39041 39058 39073 39095 39112 39131 39135 39136 39151 39156 39163 39174 39187 39190 39197 39204 39296 39299 39321 39356 39415 39475 39479 39507 39508 39598 39618 39654 39664 39698 39715 39719 39727 39731 39742 39754 39755 39763 39766 39767 39768 39772 39787 39788 39802 39823 39824 39825 39841 39854 39858 39861 39862 39881 39882 39891 39903 39918 39938 39941 39942 39946 39969 39974 39975 39992 40007 40029 40046 40065 40069 40070 40085 40090 40097 40108 40121 40124 40131 40138 40165 40198 40230 40290 40292 40321 40328 40348 40349 40408 40413 40442 40474 40496 40532 40536 40552 40564 40588 40880 40999 41055 41063 41099 41164 41224 41226 41255 41262 41282 41283 41347 41376 41408 41430 41466 41470 41486 41498 41814 41843 41933 41989 42123 42310 42464 42590 42748 42931 42967 43032 43094 43123 43130 43150 43151 43215 43244 43276 43298 43334 43338 43354 43366 43434 43499 43561 43590 43597 43617 43618 43682 43711 43743 43765 43801 43805 43806 43821 43833 43901 43966 44026 44028 44057 44064 44084 44085 44149 44178 44210 44232 44268 44272 44288 44300 44368 44401 44433 44493 44495 44524 44531 44551 44552 44611 44616 44645 44677 44699 44735 44739 44755 44767 44791 44794 44801 44835 44836 44856 44925 44959 44987 45007 45055 45075 45078 45083 45112 45144 45166 45206 45207 45222 45233 45238 45258 45266 45268 45302 45303 45323 45335 45343 45358 45395 45420 45426 45456 45465 45474 45517 45542 45545 45546 45550 45561 45570 45601 45610 45611 45628 45633 45637 45638 45646 45667 45673 45674 45677 45694 45705 45706 45725 45729 45733 45735 45742 45756 45768 45769 45770 45781 45792 45793 45795 45796 45804 45867 45874 45875 45890 45897 45923 45935 45940 45941 45945 45946 45952 45979 45986 45987 45988 45999 46037 46071 46076 46079 46083 46093 46105 46131 46134 46135 46144 46163 46167 46172 46200 46203 46209 46212 46223 46236 46237 46262 46280 46290 46315 46319 46325 46354 46360 46386 46390 46407 46408 46412 46419 46442 46446 46451 46483 46504 46506 46535 46544 46546 46562 46572 46595 46601 46630 46634 46639 46640 46662 46667 46679 46690 46694 46703 46704 46724 46729 46775 46782 46796 46855 46857 46874 46875 46879 46892 46913 46921 46922 46923 46946 46951 46960 46971 46975 46980 47017 47034 47068 47070 47074 47075 47078 47097 47101 47106 47130 47131 47134 47136 47157 47170 47171 47184 47191 47194 47196 47197 47203 47206 47221 47249 47253 47268 47288 47294 47320 47322 47324 47341 47342 47344 47346 47353 47374 47380 47385 47388 47389 47390 47393 47400 47410 47413 47414 47418 47429 47438 47447 47469 47479 47484 47496 47501 47506 47518 47535 47537 47541 47542 47545 47564 47568 47573 47574 47593 47596 47597 47598 47601 47610 47624 47637 47638 47658 47663 47709 47716 47730 47789 47791 47808 47809 47813 47826 47847 47855 47856 47857 47880 47885 47894 47905 47909 47914 47951 47968 48002 48004 48008 48009 48012 48031 48035 48040 48064 48065 48068 48070 48091 48104 48105 48118 48125 48128 48130 48131 48137 48140 48155 48183 48187 48202 48222 48228 48254 48256 48258 48275 48276 48278 48280 48287 48308 48314 48319 48322 48323 48324 48327 48334 48344 48347 48348 48352 48363 48372 48381 48403 48413 48418 48430 48435 48440 48452 48469 48471 48475 48476 48479 48498 48502 48507 48508 48527 48530 48531 48532 48535 48544 48558 48571 48572 48584 48592 48615 48618 48622 48640 48643 48661 48664 48667 48695 48697 48723 48725 48743 48745 48747 48781 48786 48791 48811 48815 48819 48839 48848 48865 48870 48880 48897 48907 48963 48969 48974 48975 48994 49002 49011 49013 49025 49908 49972 50041 50046 50051 50061 50062 50073 50098 50113 50122 50132 50144 50146 50156 50192 50212 50216 50298 50335 50360 50366 50375 50399 50412 50430 50439 50440 50453 50460 50463 50465 50466 50472 50475 50490 50518 50522 50537 50557 50563 50589 50591 50593 50610 50611 50613 50615 50622 50643 50649 50654 50657 50658 50659 50662 50669 50679 50682 50683 50687 50698 50707 50716 50738 50748 50753 50765 50770 50775 50787 50804 50806 50810 50811 50814 50833 50837 50842 50843 50862 50865 50866 50867 50870 50879 50893 50906 50907 50920 50930 50932 50939 50975 50978 50985 50989 50995 50996 51004 51007 51024 51030 51032 51046 51047 51060 51069 51077 51078 51082 51089 51103 51116 51121 51126 51134 51136 51146 51148 51149 51150 51152 51154 51162 51174 51176 51183 51205 51215 51216 51232 51235 51271 51273 51277 51278 51282 51300 51304 51305 51309 51310 51333 51337 51339 51346 51349 51360 51840 51841 51866 51892 51964 51990 51992 52012 52016 52023 52055 52060 52084 52088 52108 52166 52205 52243 52271 52294 52774 52775 52786 52800 52828 52948 52957 52980 53051 53073 53100 53133 53139 53172 53177 53178 53228 53232 53241 53242 53267 53460 53606 53635 53644 53672 53708 53709 53729 53734 53780 53787 53801 53860 53862 53879 53880 53884 53897 53918 53926 53927 53928 53951 53956 53965 53976 53980 53985 54022 54039 54073 54075 54079 54080 54083 54102 54106 54111 54135 54136 54139 54141 54162 54175 54176 54186 54194 54226 54236 54242 54247 54250 54259 54262 54264 54265 54276 54291 54299 54301 54325 54327 54329 54346 54347 54358 54359 54372 54394 54395 54399 54405 54418 54419 54421 54431 54452 54459 54474 54483 54540 54546 54547 54551 54569 54578 54601 54606 54615 54629 54633 54643 54668 54721 54796 54814 54872 54890 54944 55045 55052 55073 55096 55109 55110 55114 55121 55126 55129 55135 55143 55145 55150 55153 55155 55159 55163 55168 55174 55175 55178 55183 55185 55188 55190 55192 55198 55207 55210 55214 55215 55216 55221 55227 55230 55233 55235 55236 55237 55243 55248 55250 55251 55252 55259 55266 55267 55270 55280 55281 55283 55285 55292 55293 55309 55311 55312 55315 55316 55318 55319 55326 55328 55329 55331 55332 55333 55335 55339 55349 55353 55356 55357 55358 55361 55365 55367 55370 55372 55374 55377 55381 55386 55397 55398 55403 55408 55411 55416 55417 55419 55420 55421 55426 55433 55435 55436 55440 55441 55445 55451 55457 55462 55464 55468 55471 55472 55474 55475 55480 55481 55489 55490 55491 55497 55499 55501 55502 55503 55507 55510 55512 55513 55514 55519 55529 55530 55535 55540 55541 55543 55549 55552 55555 55558 55562 55563 55564 55567 55569 55572 55573 55574 55575 55576 55577 55578 55579 55580 55581 55582 55583 55584 55585 55586 55587 55588 55589 55590 55591 55592 55593 55594 55595 55596 55597 55598 55599 55600 55601 55602 55603 55604 55605 55606 55607 55608 55609 55610 55611 55612 55613 55614 55615 55616 55617 55618 55619 55620 55621 55622 55623 55624 55625 55626 55627 55628 55629 55630 55631 55632 55633 55634 55635 55636 55637 55638 55639 55640 55641 55642 55643 55644 55645 55646 55647 55648 55649 55650 55651 55652 55653 55654 55655 55656 55657 55658 55659 55660 55661 55662 55663 55664 55665 55666 55667 55668 55669 55670 55671 55672 55673 55674 55675 55676 55677 55678 55679 55680 55681 55682 55683 55684 55685 55686 55687 55688 55689 55690 55691 55692 55693 55694 55695 55696 55697 55698 55699 55700 55701 55702 55703 55704 55705 55706 55707 55708 55709 55710 55711 55712 55713 55714 55715 55716 55717 55718 55719 55720 55721 55722 55723 55724 55725 55726 55727 55728 55729 55730 55731 55732 55733 55734 55735 55736 55737 55738 55739 55740 55741 55742 55743 55744 55745 55746 55747 55748 55749 55750 55751 55752 55753 55754 55755 55756 55757 55758 55759 55760 55761 55762 55763 55764 55765 55766 55767 55768 55769 55770 55771 55772 55773 55774 55775 55776 55777 55778 55779 55780 55781 55782 55783 55784 55785 55786 55787 55788 55789 55790 55791 55792 55793 55794 55795 55796 55797 55798 55799 55800 55801 55802 55803 55804 55805 55806 55807 55808 55809 55810 55811 55812 55813 55814 55815 55816 55817 55818 55819 55820 55821 55822 55823 55824 55825 55826 55827 55828 55829 55830 55831 55832 55833 55834 55835 55836 55837 55838 55839 55840 55841 55842 55843 55844 55845 55846 55847 55848 55849 55850 55851 55852 55853 55854 55855 55856 55857 55858 55859 55860 55861 55862 55863 55864 55865 55866 55867 55868 55869 55870 55871 55872 55873 55874 55875 55876 55877 55878 55879 55880 55881 55882 55883 55884 55885 55886 55887 55888 55889 55890 55891 55892 55893 55894 55895 55896 55897 55898 55899 55900 55901 55902 55903 55904 55905 55906 55907 55908 55909 55910 55911 55912 55913 55914 55915 55916 55917 55918 55919 55920 55921 55922 55923 55924 55925 55926 55927 55928 55929 55930 55931 55932 55933 55934 55935 55936 55937 55938 55939 55940 55941 55942 55943 55944 55945 55946 55947 55948 55949 55950 55951 55952 55953 55954 55955 55956 55957 55958 55959 55960 55961 55963 55964 55965 55966 55967 55968 55969 55970 55971 55972 55973 55974 55975 55976 55977 55978 55979 55980 55981 55982 55983 55984 55985 55986 55987 55988 55989 55990 55991 55992 55993 55994 55995 55996 55997 55998 55999 56000 56001 56002 56003 56004 56005 56006 56007 56008 56009 56010 56011 56012 56013 56014 56015 56016 56017 56018 56019 56020 56021 56022 56023 56024 56025 56026 56027 56028 56029 56030 56031 56032 56033 56034 56035 56036 56037 56038 56039 61180 61273 61428 61551 62114 62362 63515 63516 63536 63605 63608 63658 63730 63763 63841 63913 63918 63982 63983 64003 64125 64134 64197 64230 64308 64353 64380 64385 64449 64450 64539 64573 64599 64621 64623 64717 64775 64852 64903 64917 64979 64988 64999 65000 65027 65042 65066 65068 65096 65099 65136 65140 65146 65194 65281 65310 65319 65383 65384 65396 65397 65407 65409 65427 65435 65455 65464 65466 65473 65507 65509 65533 65535 65537 65555 65566 65567 65583 65586 65598 65603 65606 65613 65627 65682 65709 65748 65765 65775 65777 65786 65810 65837 65838 65850 65851 65864 65874 65875 65876 65894 65922 65933 65940 65962 65971 65972 65974 65976 66000 66002 66004 66022 66026 66030 66033 66034 66053 66065 66070 66073 66080 66094 66118 66149 66176 66215 66221 66232 66242 66244 66248 66253 66277 66281 66304 66305 66317 66318 66489 66720 66784 66805 66835 66836 66853 66863 66874 66877 66934 66938 66944 66980 67032 67211 67250 67251 67252 67255 67256 67262 67264 67265 67268 67270 67271 67272 67275 67276 67277 67278 67281 67282 67283 67284 67285 67287 67290 67292 67295 67298 67299 67301 67302 67303 67304 67305 67307 67308 67309 67312 67313 67316 67317 67318 67319 67320 67322 67323 67324 67325 67326 67327 67329 67330 67332 67333 67334 67335 67339 67340 67341 67343 67344 67349 67350 67351 67352 67353 67354 67355 67357 67358 67359 67363 67365 67367 67369 67371 67372 67373 67375 67376 67377 67378 67379 67382 67385 67389 67390 67391 67392 67393 67394 67397 67401 67403 67404 67405 67407 67408 67410 67411 67414 67417 67423 67425 67427 67428 67434 67435 67438 67439 67441 67443 67447 67448 67452 67453 67454 67457 67458 67460 67461 67468 67469 67470 67471 67474 67475 67479 67480 67481 67484 67485 67489 67490 67491 67493 67494 67495 67497 67498 67499 67501 67503 67504 67507 67509 67510 67512 67519 67520 67521 67522 67524 67525 67526 67527 67528 67531 67535 67536 67540 67544 67545 67550 67551 67553 67554 67558 67559 67560 67561 67562 67563 67565 67568 67573 67575 67576 67580 67583 67586 67587 67588 67593 67595 67596 67598 67599 67601 67602 67607 67609 67610 67611 67614 67616 67617 67618 67619 67620 67622 67623 67626 67627 67630 67633 67635 67638 67639 67640 67641 67642 67643 67645 67649 67650 67651 67652 67654 67655 67661 67662 67663 67667 67670 67671 67672 67674 67675 67677 67678 67682 67683 67684 67685 67686 67687 67691 67692 67693 67694 67700 67701 67703 67705 67706 67707 67709 67711 67714 67718 67719 67739 67742 67744 67745 67751 67762 67769 67770 67797 67801 67842 67868 67870 67871 67872 67890 67892 67894 67895 67901 67907 67914 67922 67928 67933 67936 67937 67938 67941 67948 67961 67962 67966 67977 67986 67990 67995 68017 68027 68044 68049 68054 68083 68089 68090 68093 68112 68116 68121 68122 68145 68149 68151 68158 68172 68653 68823 69017 69209 69367 69396 69450 69522 70053 70054 70077 70079 70171 70225 70227 70229 70271 70272 70276 70277 70283 70321 70340 70342 70367 70415 70418 70428 70451 70456 70460 70484 70507 70520 70521 70532 70546 70547 70674 70691 70692 70694 70696 70703 70738 70739 70743 70744 70750 70768 70788 70797 70832 70834 70882 70885 70914 70918 70923 70947 70951 70974 70987 70988 70994 70999 71001 71010 71011 71022 71023 71030 71040 71041 71070 71085 71096 71105 71108 71125 71141 71144 71153 71154 71158 71159 71161 71162 71163 71165 71170 71193 71196 71197 71205 71206 71218 71235 71246 71256 71260 71267 71272 71289 71291 71294 71297 71301 71311 71312 71315 71318 71321 71322 71327 71334 71344 71345 71349 71352 71369 71385 71390 71394 71417 71418 71421 71424 71430 71433 71436 71441 71442"
indices_12_1_GATGCN = "4,174,179,222,317,406,1807,4360,4609,8410,8492,8499,8563,8580,8581,8609,8628,8653,8723,8774,8812,8840,8863,9279,9307,12299,12612,12613,12636,12704,12717,12734,12766,12783,12788,12831,12860,12926,12936,12952,12977,13015,13043,15101,15881,15882,15905,15908,15917,15979,16033,16035,16053,16056,16057,16058,16064,16098,16100,16111,16149,16193,16195,16205,16217,16243,16246,16257,16275,16279,16284,16308,16312,16315,16335,16348,16349,16372,16374,16375,16440,16453,16454,16455,16469,16502,16524,16525,16565,16566,16567,16616,16660,16662,16672,16684,16710,16713,16724,16742,16746,16751,16775,16779,16791,16802,18217,18387,18610,18619,32227,32378,32405,32694,32847,32864,33058,33096,33627,33656,33717,33731,33752,33770,33811,33871,33875,33903,33904,33994,33998,34026,34050,34060,34184,34525,34928,34984,35153,35188,35305,35395,35495,35524,35585,35620,35679,35739,35743,35771,35772,35862,35866,35894,35918,36896,37144,37363,37488,37523,37611,37730,37786,38297,38314,38318,38326,38330,38341,38353,38354,38362,38365,38366,38367,38371,38386,38387,38401,38422,38423,38424,38440,38457,38460,38461,38480,38481,38490,38502,38517,38537,38540,38541,38545,38573,38574,38591,38606,38628,38645,38664,38668,38669,38684,38689,38696,38707,38720,38723,38730,38737,38829,38832,38854,38889,38948,39008,39012,39040,39041,39131,39151,39187,39197,39231,39248,39252,39260,39264,39275,39287,39288,39296,39299,39300,39301,39305,39320,39321,39335,39356,39357,39358,39374,39387,39391,39394,39395,39414,39415,39424,39436,39451,39471,39474,39475,39479,39502,39507,39508,39525,39540,39562,39579,39598,39602,39603,39618,39623,39630,39641,39654,39657,39664,39671,39698,39731,39763,39823,39825,39854,39861,39881,39882,39941,39946,39975,40007,40029,40065,40069,40085,40097,40121,40413,40532,40588,40596,40632,40697,40757,40759,40788,40795,40815,40816,40880,40909,40941,40963,40999,41003,41019,41031,41347,41376,41466,41522,41656,41843,41997,42123,42281,42464,42500,42565,42627,42656,42663,42683,42684,42748,42777,42809,42831,42867,42871,42887,42899,42967,43032,43094,43123,43130,43150,43151,43215,43244,43276,43298,43334,43338,43339,43354,43366,43434,43499,43559,43561,43590,43597,43617,43618,43682,43711,43743,43765,43801,43805,43821,43833,43901,43934,43966,44026,44028,44057,44064,44084,44085,44144,44149,44178,44210,44232,44268,44272,44288,44300,44324,44327,44334,44368,44369,44389,44458,44492,44520,44540,44588,44608,44611,44616,44645,44677,44699,44739,44740,44755,44766,44771,44791,44799,44801,44835,44836,44856,44868,44876,44891,44928,44953,44959,44989,44998,45007,45050,45075,45078,45079,45083,45094,45103,45134,45143,45144,45161,45166,45170,45171,45179,45200,45206,45207,45210,45227,45238,45239,45258,45262,45266,45268,45275,45289,45301,45302,45303,45314,45325,45326,45328,45329,45337,45400,45407,45408,45423,45430,45456,45468,45473,45474,45478,45479,45485,45512,45519,45520,45521,45532,45570,45604,45609,45612,45616,45626,45638,45664,45667,45668,45677,45696,45700,45705,45733,45736,45742,45745,45756,45769,45770,45795,45813,45823,45848,45852,45858,45887,45893,45919,45923,45940,45941,45945,45952,45975,45979,45984,46016,46037,46039,46068,46077,46079,46095,46105,46128,46134,46163,46167,46172,46173,46195,46200,46212,46223,46227,46236,46237,46257,46262,46308,46315,46329,46388,46390,46407,46408,46412,46425,46446,46454,46455,46456,46479,46484,46493,46504,46508,46513,46550,46567,46601,46603,46607,46608,46611,46630,46634,46639,46663,46664,46667,46669,46690,46703,46704,46724,46729,46775,46782,46796,46855,46857,46874,46875,46879,46892,46913,46921,46922,46923,46946,46951,46960,46971,46975,46980,47017,47034,47068,47070,47074,47075,47078,47097,47101,47106,47130,47131,47134,47136,47157,47170,47171,47184,47191,47194,47196,47197,47203,47206,47221,47249,47253,47268,47288,47294,47320,47322,47324,47341,47342,47344,47346,47353,47374,47380,47385,47388,47389,47390,47393,47400,47410,47413,47414,47418,47429,47438,47447,47469,47479,47484,47496,47501,47506,47518,47535,47537,47541,47542,47545,47564,47568,47573,47574,47593,47596,47597,47598,47601,47610,47624,47637,47638,47650,47658,47681,47684,47688,47706,47709,47727,47730,47733,47761,47763,47789,47791,47809,47811,47813,47847,47852,47857,47877,47881,47885,47905,47914,47931,47936,47946,47963,47973,48029,48035,48040,48041,48060,48068,48077,48079,48091,48974,49038,49107,49112,49117,49127,49128,49139,49164,49179,49188,49198,49210,49212,49222,49258,49278,49282,49364,49401,49426,49432,49441,49465,49478,49496,49505,49506,49519,49526,49529,49531,49532,49538,49541,49556,49584,49588,49603,49623,49629,49655,49657,49659,49676,49677,49679,49681,49688,49709,49715,49720,49723,49724,49725,49728,49735,49745,49748,49749,49753,49764,49773,49782,49804,49814,49819,49831,49836,49841,49853,49870,49872,49876,49877,49880,49899,49903,49908,49909,49928,49931,49932,49933,49936,49945,49959,49972,49973,49986,49996,49998,50005,50041,50044,50051,50055,50061,50062,50070,50073,50090,50096,50098,50112,50113,50126,50135,50143,50144,50148,50155,50169,50182,50187,50192,50200,50202,50212,50214,50215,50216,50218,50220,50228,50240,50242,50249,50271,50281,50282,50298,50301,50337,50339,50343,50344,50348,50366,50370,50371,50375,50376,50399,50403,50405,50412,50415,50426,50906,50907,50932,50958,51030,51056,51058,51078,51082,51089,51121,51126,51150,51154,51174,51232,51271,51309,51337,51360,51840,51841,51852,51866,51894,52014,52023,52046,52117,52139,52166,52199,52205,52238,52243,52244,52294,52298,52307,52308,52333,52526,52672,52701,52710,52738,52774,52775,52785,52793,52825,52835,52841,52846,52849,52858,52861,52863,52864,52875,52890,52898,52900,52924,52926,52928,52945,52946,52957,52958,52971,52993,52994,52998,53004,53017,53018,53020,53030,53051,53058,53073,53082,53139,53145,53146,53150,53168,53177,53200,53205,53214,53228,53232,53242,53267,53320,53395,53413,53471,53489,53543,53644,53651,53672,53695,53708,53709,53713,53720,53725,53728,53734,53742,53744,53749,53752,53754,53758,53762,53767,53773,53774,53777,53782,53784,53787,53789,53791,53797,53806,53809,53813,53814,53815,53820,53826,53829,53832,53834,53835,53836,53842,53847,53849,53850,53851,53858,53865,53866,53869,53879,53880,53882,53884,53891,53892,53908,53910,53911,53914,53915,53917,53918,53925,53927,53928,53930,53931,53932,53934,53938,53948,53952,53955,53956,53957,53960,53964,53966,53969,53971,53973,53976,53980,53985,53996,53997,54002,54007,54010,54015,54016,54018,54019,54020,54025,54032,54034,54035,54039,54040,54044,54050,54056,54061,54063,54067,54070,54071,54073,54074,54079,54080,54088,54089,54090,54096,54098,54100,54101,54102,54106,54109,54111,54112,54113,54118,54128,54129,54134,54139,54140,54142,54148,54151,54154,54157,54161,54162,54163,54166,54168,54171,54172,54173,54174,54175,54176,54177,54178,54179,54180,54181,54182,54183,54184,54185,54186,54187,54188,54189,54190,54191,54192,54193,54194,54195,54196,54197,54198,54199,54200,54201,54202,54203,54204,54205,54206,54207,54208,54209,54210,54211,54212,54213,54214,54215,54216,54217,54218,54219,54220,54221,54222,54223,54224,54225,54226,54227,54228,54229,54230,54231,54232,54233,54234,54235,54236,54237,54238,54239,54240,54241,54242,54243,54244,54245,54246,54247,54248,54249,54250,54251,54252,54253,54254,54255,54256,54257,54258,54259,54260,54261,54262,54263,54264,54265,54266,54267,54268,54269,54270,54271,54272,54273,54274,54275,54276,54277,54278,54279,54280,54281,54282,54283,54284,54285,54286,54287,54288,54289,54290,54291,54292,54293,54294,54295,54296,54297,54298,54299,54300,54301,54302,54303,54304,54305,54306,54307,54308,54309,54310,54311,54312,54313,54314,54315,54316,54317,54318,54319,54320,54321,54322,54323,54324,54325,54326,54327,54328,54329,54330,54331,54332,54333,54334,54335,54336,54337,54338,54339,54340,54341,54342,54343,54344,54345,54346,54347,54348,54349,54350,54351,54352,54353,54354,54355,54356,54357,54358,54359,54360,54361,54362,54363,54364,54365,54366,54367,54368,54369,54370,54371,54372,54373,54374,54375,54376,54377,54378,54379,54380,54381,54382,54383,54384,54385,54386,54387,54388,54389,54390,54391,54392,54393,54394,54395,54396,54397,54398,54399,54400,54401,54402,54403,54404,54405,54406,54407,54408,54409,54410,54411,54412,54413,54414,54415,54416,54417,54418,54419,54420,54421,54422,54423,54424,54425,54426,54427,54428,54429,54430,54431,54432,54433,54434,54435,54436,54437,54438,54439,54440,54441,54442,54443,54444,54445,54446,54447,54448,54449,54450,54451,54452,54453,54454,54455,54456,54457,54458,54459,54460,54461,54462,54463,54464,54465,54466,54467,54468,54469,54470,54471,54472,54473,54474,54475,54476,54477,54478,54479,54480,54481,54482,54483,54484,54485,54486,54487,54488,54489,54490,54491,54492,54493,54494,54495,54496,54497,54498,54499,54500,54501,54502,54503,54504,54505,54506,54507,54508,54509,54510,54511,54512,54513,54514,54515,54516,54517,54518,54519,54520,54521,54522,54523,54524,54525,54526,54527,54528,54529,54530,54531,54532,54533,54534,54535,54536,54537,54538,54539,54540,54541,54542,54543,54544,54545,54546,54547,54548,54549,54550,54551,54552,54553,54554,54555,54556,54557,54558,54559,54560,54562,54563,54564,54565,54566,54567,54568,54569,54570,54571,54572,54573,54574,54575,54576,54577,54578,54579,54580,54581,54582,54583,54584,54585,54586,54587,54588,54589,54590,54591,54592,54593,54594,54595,54596,54597,54598,54599,54600,54601,54602,54603,54604,54605,54606,54607,54608,54609,54610,54611,54612,54613,54614,54615,54616,54617,54618,54619,54620,54621,54622,54623,54624,54625,54626,54627,54628,54629,54630,54631,54632,54633,54634,54635,54636,54637,54638,59779,59872,60027,60150,60713,60961,62114,62115,62135,62204,62207,62257,62329,62362,62440,62512,62517,62581,62582,62602,62724,62733,62796,62829,62907,62952,62979,62984,63048,63049,63138,63172,63198,63220,63222,63316,63374,63451,63502,63516,63578,63587,63598,63599,63626,63641,63665,63667,63695,63698,63735,63739,63745,63793,63880,63909,63918,63982,63983,63995,63996,64006,64008,64026,64034,64054,64063,64065,64072,64106,64108,64132,64134,64136,64154,64165,64166,64182,64185,64197,64202,64205,64212,64226,64281,64308,64347,64364,64374,64376,64385,64409,64436,64437,64449,64450,64463,64473,64474,64475,64493,64521,64532,64539,64561,64570,64571,64573,64575,64599,64601,64603,64621,64625,64629,64632,64633,64652,64664,64669,64672,64679,64693,64717,64748,64775,64814,64820,64831,64841,64843,64847,64852,64876,64880,64903,64904,64916,64917,65088,65319,65383,65404,65434,65435,65452,65462,65473,65476,65533,65537,65543,65579,65631,65810,65849,65850,65851,65854,65855,65861,65863,65864,65867,65869,65870,65871,65874,65875,65876,65877,65880,65881,65882,65883,65884,65886,65889,65891,65894,65897,65898,65900,65901,65902,65903,65904,65906,65907,65908,65911,65912,65915,65916,65917,65918,65919,65921,65922,65923,65924,65925,65926,65928,65929,65931,65932,65933,65934,65938,65939,65940,65942,65943,65948,65949,65950,65951,65952,65953,65954,65956,65957,65958,65962,65964,65966,65968,65970,65971,65972,65974,65975,65976,65977,65978,65981,65984,65988,65989,65990,65991,65992,65993,65996,66000,66002,66003,66004,66006,66007,66009,66010,66013,66016,66022,66024,66026,66027,66033,66034,66037,66038,66040,66042,66046,66047,66051,66052,66053,66056,66057,66059,66060,66067,66068,66069,66070,66073,66074,66078,66079,66080,66083,66084,66088,66089,66090,66092,66093,66094,66096,66097,66098,66100,66102,66103,66106,66108,66109,66111,66118,66119,66120,66121,66123,66124,66125,66126,66127,66130,66134,66135,66139,66143,66144,66149,66150,66152,66153,66157,66158,66159,66160,66161,66162,66164,66167,66172,66174,66175,66179,66182,66185,66186,66187,66192,66194,66195,66197,66198,66200,66201,66206,66208,66209,66210,66213,66215,66216,66217,66218,66219,66221,66222,66225,66226,66229,66232,66234,66237,66238,66239,66240,66241,66242,66244,66248,66249,66250,66251,66253,66254,66260,66261,66262,66266,66269,66270,66271,66273,66274,66276,66277,66281,66282,66283,66284,66285,66286,66290,66291,66292,66293,66299,66300,66302,66304,66305,66306,66308,66310,66313,66317,66318,66338,66341,66343,66344,66350,66361,66368,66369,66396,66400,66441,66467,66469,66470,66471,66489,66491,66493,66494,66500,66506,66513,66521,66527,66532,66535,66536,66537,66540,66547,66560,66561,66565,66576,66585,66589,66594,66616,66626,66643,66648,66653,66682,66688,66689,66692,66711,66715,66720,66721,66744,66748,66750,66757,66771,67252,67422,67616,67808,67966,67995,68049,68121,68652,68653,68676,68678,68770,68824,68826,68828,68870,68871,68875,68876,68882,68920,68939,68941,68966,69014,69017,69027,69050,69055,69059,69083,69106,69119,69120,69131,69145,69146,69273,69290,69291,69293,69295,69302,69337,69338,69342,69343,69349,69367,69387,69396,69431,69433,69481,69484,69513,69517,69522,69546,69550,69573,69586,69587,69593,69598,69600,69609,69610,69621,69622,69629,69639,69640,69669,69684,69695,69704,69707,69724,69740,69743,69752,69753,69757,69758,69760,69761,69762,69764,69769,69792,69795,69796,69804,69805,69817,69834,69845,69855,69859,69866,69871,69888,69890,69893,69896,69900,69910,69911,69914,69917,69920,69921,69926,69933,69943,69944,69948,69951,69968,69984,69989,69993,70016,70017,70020,70023,70029,70032,70035,70040,70041"
indices_14_0_GATGCN = "55642 55645 55648 55651 55658 55665 55669 55680 55702 55716 55717 55719 55730 55735 55736 55760 55818 55939 55948"
indices_13_0_GATGCN = "46172 46236 46639 47171 47573 48105 48507 48572 48786 50440 50842 51309 55573 55574 55575 55576 55577 55578 55579 55580 55581 55582 55583 55584 55585 55586 55587 55588 55589 55590 55591 55593 55595 55596 55597 55598 55599 55600 55601 55602 55603 55604 55605 55606 55607 55608 55609 55610 55611 55612 55613 55614 55615 55616 55617 55618 55619 55620 55621 55622 55623 55624 55625 55626 55627 55628 55629 55630 55631 55632 55633 55634 55635 55636 55637 55638 55639 55640 55641 55642 55643 55644 55645 55646 55647 55648 55649 55650 55651 55652 55653 55654 55655 55656 55657 55658 55659 55660 55661 55662 55664 55665 55666 55667 55668 55669 55670 55671 55672 55674 55675 55676 55677 55679 55680 55681 55682 55683 55684 55685 55686 55687 55688 55689 55690 55692 55693 55694 55696 55697 55698 55699 55700 55701 55702 55703 55704 55705 55706 55707 55708 55709 55711 55712 55713 55714 55715 55716 55717 55718 55719 55720 55721 55722 55723 55724 55725 55726 55727 55728 55729 55730 55731 55732 55733 55734 55735 55736 55737 55738 55739 55741 55742 55743 55744 55745 55748 55749 55750 55751 55752 55753 55754 55756 55757 55758 55759 55760 55761 55762 55763 55764 55766 55767 55769 55770 55771 55772 55773 55774 55776 55778 55779 55780 55781 55782 55783 55784 55785 55786 55787 55788 55789 55790 55791 55792 55793 55795 55796 55797 55798 55799 55800 55801 55802 55803 55804 55805 55806 55807 55808 55809 55810 55811 55813 55814 55815 55816 55817 55818 55819 55820 55821 55822 55823 55824 55825 55826 55828 55829 55831 55832 55834 55835 55836 55837 55838 55839 55840 55842 55843 55844 55845 55846 55847 55849 55850 55851 55852 55853 55854 55855 55856 55857 55858 55859 55860 55861 55862 55863 55864 55865 55866 55867 55868 55869 55870 55872 55873 55874 55875 55876 55878 55879 55880 55881 55882 55883 55884 55885 55886 55887 55888 55889 55890 55891 55893 55894 55895 55897 55898 55899 55900 55901 55903 55904 55905 55906 55908 55909 55910 55911 55912 55913 55914 55915 55916 55917 55918 55919 55920 55921 55922 55923 55924 55925 55926 55927 55928 55929 55930 55931 55932 55933 55934 55935 55936 55937 55938 55939 55941 55942 55943 55944 55945 55947 55948 55949 55950 55951 55952 55953 55955 55956 55957 55958 55959 55960 55961 55963 55964 55965 55966 55967 55968 55969 55970 55971 55972 55973 55974 55975 55976 55977 55978 55979 55980 55981 55982 55983 55986 55987 55988 55989 55990 55991 55992 55993 55994 55995 55996 55997 55998 55999 56000 56001 56002 56003 56005 56006 56007 56008 56009 56010 56011 56012 56013 56014 56015 56016 56017 56018 56019 56020 56021 56022 56023 56024 56025 56026 56027 56028 56029 56030 56031 56032 56034 56035 56036 56037 56038 56039 65384 65533 65851 66253 66877 66934 67211 67252 67284 67317 67320 67323 67326 67340 67344 67352 67358 67405 67411 67435 67491 67495 67497 67507 67614 67623 67654 67718 67719 68121 68172"
indices_13_5_GATGCN = "55577 55580 55587 55589 55590 55603 55605 55610 55627 55630 55632 55633 55636 55638 55642 55643 55644 55645 55646 55647 55648 55649 55650 55651 55655 55658 55659 55660 55662 55665 55666 55669 55677 55680 55683 55684 55688 55697 55700 55701 55702 55703 55705 55713 55714 55716 55717 55719 55726 55728 55730 55735 55736 55739 55748 55759 55760 55771 55772 55780 55781 55793 55796 55805 55806 55810 55815 55816 55818 55820 55824 55831 55834 55852 55853 55873 55875 55885 55890 55906 55913 55921 55939 55941 55948 55956 55970 55972 55976 55978 55979 56007 56016 56018 56024 56032 66934"
# list1 = indices_12_4_GCNFG
# list2 = indices_12_4_GCNGATFG
# common_elements = extract_common_elements(list1, list2)
# print(common_elements)

input_file = "./test.txt"  # 输入文件名
normal_pairs = "result_elements_12_1_GATGCN.txt"
common_pairs = "common_result_elements.txt"
output_file = normal_pairs  # 输出文件名

# 读取输入文件的内容
with open(input_file, "r") as f:
    lines = f.readlines()

# 提取对应索引的行
# output_lines = [lines[i] for i in indices_12_7_GCNGATFG]
# output_lines = [lines[i] for i in common_elements]
output_lines = [lines[int(i)] for i in indices_12_1_GATGCN.split(',')]

# 写入输出文件
with open(output_file, "w") as f:
    f.writelines(output_lines)

