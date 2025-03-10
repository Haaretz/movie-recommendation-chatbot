@Library('jenkins-lib') _
simpleBuild{
    argoApp = "app-of-apps"
    autodeploy = true
    triggers = [pollSCM('H/3 * * * *')]
}
