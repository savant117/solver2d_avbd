// SPDX-FileCopyrightText: 2024 Erin Catto
// SPDX-License-Identifier: MIT

#define _CRT_SECURE_NO_WARNINGS
// #define IMGUI_DISABLE_OBSOLETE_FUNCTIONS 1

#if defined(_WIN32)
	#include <crtdbg.h>

	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif

	#include <windows.h>
#endif

#include "draw.h"
#include "sample.h"
#include "settings.h"

#include "solver2d/constants.h"
#include "solver2d/math.h"
#include "solver2d/timer.h"

#include <glad/glad.h>
// Keep glad.h before glfw3.h
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdio.h>
#include <stdlib.h>

GLFWwindow* g_mainWindow = nullptr;
static int s_selection = 0;
static Sample* s_samples[s2_solverTypeCount];
static Settings s_settings;
static bool s_rightMouseDown = false;
static s2Vec2 s_clickPointWS = s2Vec2_zero;
static float s_windowScale = 1.0f;
static float s_framebufferScale = 1.0f;

void glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "GLFW error occurred. Code: %d. Description: %s\n", error, description);
}

static inline int CompareSamples(const void* a, const void* b)
{
	SampleEntry* sa = (SampleEntry*)a;
	SampleEntry* sb = (SampleEntry*)b;

	int result = strcmp(sa->category, sb->category);
	if (result == 0)
	{
		result = strcmp(sa->name, sb->name);
	}

	return result;
}

static void SortTests()
{
	qsort(g_sampleEntries, g_sampleCount, sizeof(SampleEntry), CompareSamples);
}

static void RestartTest()
{
	s_settings.sampleIndex = s_selection;
	for (int i = 0; i < s2_solverTypeCount; ++i)
	{
		if (s_samples[i] != nullptr)
		{
			delete s_samples[i];
			s_samples[i] = nullptr;
		}
	}

	s_settings.restart = true;

	for (int i = 0; i < s2_solverTypeCount; ++i)
	{
		if (s_settings.enabledSolvers[i])
		{
			s_samples[i] = g_sampleEntries[s_settings.sampleIndex].createFcn(s_settings, s2SolverType(i));
		}
	}
}

static void CreateUI(GLFWwindow* window, const char* glslVersion)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	bool success;
	success = ImGui_ImplGlfw_InitForOpenGL(window, false);
	if (success == false)
	{
		printf("ImGui_ImplGlfw_InitForOpenGL failed\n");
		assert(false);
	}

	success = ImGui_ImplOpenGL3_Init(glslVersion);
	if (success == false)
	{
		printf("ImGui_ImplOpenGL3_Init failed\n");
		assert(false);
	}

	// Search for font file
	const char* fontPath1 = "data/droid_sans.ttf";
	const char* fontPath2 = "../data/droid_sans.ttf";
	const char* fontPath = nullptr;
	FILE* file1 = fopen(fontPath1, "rb");
	FILE* file2 = fopen(fontPath2, "rb");
	if (file1)
	{
		fontPath = fontPath1;
		fclose(file1);
	}

	if (file2)
	{
		fontPath = fontPath2;
		fclose(file2);
	}

	if (fontPath)
	{
		ImFontConfig fontConfig;
		fontConfig.RasterizerMultiply = s_windowScale * s_framebufferScale;
		ImGui::GetIO().Fonts->AddFontFromFileTTF(fontPath, 14.0f, &fontConfig);
	}
}

static void DestroyUI()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

static void ResizeWindowCallback(GLFWwindow*, int width, int height)
{
	g_camera.m_width = int(width / s_windowScale);
	g_camera.m_height = int(height / s_windowScale);
	s_settings.windowWidth = int(width / s_windowScale);
	s_settings.windowHeight = int(height / s_windowScale);
}

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
	if (ImGui::GetIO().WantCaptureKeyboard)
	{
		return;
	}

	if (action == GLFW_PRESS)
	{
		switch (key)
		{
			case GLFW_KEY_ESCAPE:
				// Quit
				glfwSetWindowShouldClose(g_mainWindow, GL_TRUE);
				break;

			case GLFW_KEY_LEFT:
				g_camera.m_center.x -= 0.5f;
				break;

			case GLFW_KEY_RIGHT:
				g_camera.m_center.x += 0.5f;
				break;

			case GLFW_KEY_DOWN:
				g_camera.m_center.y -= 0.5f;
				break;

			case GLFW_KEY_UP:
				g_camera.m_center.y += 0.5f;
				break;

			case GLFW_KEY_HOME:
				g_camera.ResetView();
				break;

			case GLFW_KEY_R:
				RestartTest();
				break;

			case GLFW_KEY_P:
				s_settings.pause = !s_settings.pause;
				break;

			case GLFW_KEY_LEFT_BRACKET:
				// Switch to previous test
				--s_selection;
				if (s_selection < 0)
				{
					s_selection = g_sampleCount - 1;
				}
				break;

			case GLFW_KEY_RIGHT_BRACKET:
				// Switch to next test
				++s_selection;
				if (s_selection == g_sampleCount)
				{
					s_selection = 0;
				}
				break;

			case GLFW_KEY_9:
				s_settings.primaryIterations = S2_MAX(1, s_settings.primaryIterations - 1);
				break;

			case GLFW_KEY_0:
				s_settings.primaryIterations += 1;
				break;

			case GLFW_KEY_COMMA:
				// Switch to previous solver
				{
					int index = -1;
					for (int i = 0; i < s2_solverTypeCount; ++i)
					{
						if (s_settings.enabledSolvers[i] == true)
						{
							index = i;
							s_settings.enabledSolvers[i] = false;
						}
					}

					index = index > 0 ? index - 1 : s2_solverTypeCount - 1;
					s_settings.enabledSolvers[index] = true;
					RestartTest();
				}
				break;

			case GLFW_KEY_PERIOD:
				// Switch to next solver
				{
					int index = -1;
					for (int i = 0; i < s2_solverTypeCount; ++i)
					{
						if (s_settings.enabledSolvers[i] == true)
						{
							index = i;
							s_settings.enabledSolvers[i] = false;
						}
					}

					index = index < s2_solverTypeCount - 1 ? index + 1 : 0;
					s_settings.enabledSolvers[index] = true;
					RestartTest();
				}
				break;

			case GLFW_KEY_TAB:
				g_draw.m_showUI = !g_draw.m_showUI;
				break;

			default:
				for (int i = 0; i < s2_solverTypeCount; ++i)
				{
					if (s_samples[i] != nullptr)
					{
						s_samples[i]->Keyboard(key);
					}
				}
				break;
		}
	}
	else if (action == GLFW_RELEASE)
	{
		for (int i = 0; i < s2_solverTypeCount; ++i)
		{
			if (s_samples[i] != nullptr)
			{
				s_samples[i]->KeyboardUp(key);
			}
		}
	}
}

static void CharCallback(GLFWwindow* window, unsigned int c)
{
	ImGui_ImplGlfw_CharCallback(window, c);
}

static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

	if (ImGui::GetIO().WantCaptureMouse)
	{
		return;
	}

	double xd, yd;
	glfwGetCursorPos(g_mainWindow, &xd, &yd);
	s2Vec2 ps = {float(xd) / s_windowScale, float(yd) / s_windowScale};

	// Use the mouse to move things around.
	if (button == GLFW_MOUSE_BUTTON_1)
	{
		s2Vec2 pw = g_camera.ConvertScreenToWorld(ps);
		if (action == GLFW_PRESS)
		{
			for (int i = 0; i < s2_solverTypeCount; ++i)
			{
				if (s_samples[i] != nullptr)
				{
					s_samples[i]->MouseDown(pw, button, mods);
				}
			}
		}

		if (action == GLFW_RELEASE)
		{
			for (int i = 0; i < s2_solverTypeCount; ++i)
			{
				if (s_samples[i] != nullptr)
				{
					s_samples[i]->MouseUp(pw, button);
				}
			}
		}
	}
	else if (button == GLFW_MOUSE_BUTTON_2)
	{
		if (action == GLFW_PRESS)
		{
			s_clickPointWS = g_camera.ConvertScreenToWorld(ps);
			s_rightMouseDown = true;
		}

		if (action == GLFW_RELEASE)
		{
			s_rightMouseDown = false;
		}
	}
}

static void MouseMotionCallback(GLFWwindow* window, double xd, double yd)
{
	s2Vec2 ps = {float(xd) / s_windowScale, float(yd) / s_windowScale};

	ImGui_ImplGlfw_CursorPosCallback(window, ps.x, ps.y);

	s2Vec2 pw = g_camera.ConvertScreenToWorld(ps);
	for (int i = 0; i < s2_solverTypeCount; ++i)
	{
		if (s_samples[i] != nullptr)
		{
			s_samples[i]->MouseMove(pw);
		}
	}

	if (s_rightMouseDown)
	{
		s2Vec2 diff = s2Sub(pw, s_clickPointWS);
		g_camera.m_center.x -= diff.x;
		g_camera.m_center.y -= diff.y;
		s_clickPointWS = g_camera.ConvertScreenToWorld(ps);
	}
}

static void ScrollCallback(GLFWwindow* window, double dx, double dy)
{
	ImGui_ImplGlfw_ScrollCallback(window, dx, dy);
	if (ImGui::GetIO().WantCaptureMouse)
	{
		return;
	}

	if (dy > 0)
	{
		g_camera.m_zoom /= 1.1f;
	}
	else
	{
		g_camera.m_zoom *= 1.1f;
	}
}

static void UpdateUI(s2Color* solverColors)
{
	float menuWidth = 200.0f;
	if (g_draw.m_showUI == false)
	{
		return;
	}

	ImGui::SetNextWindowPos({g_camera.m_width - menuWidth - 10.0f, 10.0f});
	ImGui::SetNextWindowSize({menuWidth, g_camera.m_height - 20.0f});

	ImGui::Begin("Tools", &g_draw.m_showUI, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

	if (ImGui::BeginTabBar("ControlTabs", ImGuiTabBarFlags_None))
	{
		if (ImGui::BeginTabItem("Controls"))
		{
			ImGui::PushItemWidth(100.0f);
			ImGui::SliderInt("Main Iters", &s_settings.primaryIterations, 0, 50);
			ImGui::SliderInt("Extra Iters", &s_settings.secondaryIterations, 0, 50);
			ImGui::SliderInt("Multi-Steps", &s_settings.multiSteps, 1, 200);
			ImGui::SliderFloat("Hertz", &s_settings.hertz, 5.0f, 480.0f, "%.0f Hz");
			ImGui::Checkbox("Warm Starting", &s_settings.enableWarmStarting);
			ImGui::PopItemWidth();

			ImGui::Separator();

			s2Color c = solverColors[s2_solverPGS];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("PGS", &s_settings.enabledSolvers[s2_solverPGS]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverPGS_NGS];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("PGS NGS", &s_settings.enabledSolvers[s2_solverPGS_NGS]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverPGS_NGS_Block];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("PGS NGS Block", &s_settings.enabledSolvers[s2_solverPGS_NGS_Block]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverPGS_Soft];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("PGS Soft", &s_settings.enabledSolvers[s2_solverPGS_Soft]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverSoftStep];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("Soft Step", &s_settings.enabledSolvers[s2_solverSoftStep]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverTGS_Sticky];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("TGS Sticky", &s_settings.enabledSolvers[s2_solverTGS_Sticky]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverTGS_Soft];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("TGS Soft", &s_settings.enabledSolvers[s2_solverTGS_Soft]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverTGS_NGS];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("TGS NGS", &s_settings.enabledSolvers[s2_solverTGS_NGS]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverXPBD];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("XPBD", &s_settings.enabledSolvers[s2_solverXPBD]);
			ImGui::PopStyleColor();

			c = solverColors[s2_solverAVBD];
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{c.r, c.g, c.b, c.a});
			ImGui::Checkbox("AVBD", &s_settings.enabledSolvers[s2_solverAVBD]);
			ImGui::PopStyleColor();


			ImGui::Separator();

			ImGui::Checkbox("Shapes", &s_settings.drawShapes);
			ImGui::Checkbox("Joints", &s_settings.drawJoints);
			ImGui::Checkbox("AABBs", &s_settings.drawAABBs);
			ImGui::Checkbox("Contact Points", &s_settings.drawContactPoints);
			ImGui::Checkbox("Contact Normals", &s_settings.drawContactNormals);
			ImGui::Checkbox("Contact Impulses", &s_settings.drawContactImpulse);
			ImGui::Checkbox("Friction Impulses", &s_settings.drawFrictionImpulse);
			ImGui::Checkbox("Center of Masses", &s_settings.drawMass);
			ImGui::Checkbox("Statistics", &s_settings.drawStats);

			ImVec2 button_sz = ImVec2(-1, 0);
			if (ImGui::Button("Pause (P)", button_sz))
			{
				s_settings.pause = !s_settings.pause;
			}

			if (ImGui::Button("Single Step (O)", button_sz))
			{
				s_settings.singleStep = !s_settings.singleStep;
			}

			if (ImGui::Button("Restart (R)", button_sz))
			{
				RestartTest();
			}

			if (ImGui::Button("Quit", button_sz))
			{
				glfwSetWindowShouldClose(g_mainWindow, GL_TRUE);
			}

			ImGui::EndTabItem();
		}

		ImGuiTreeNodeFlags leafNodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
		leafNodeFlags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

		ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

		if (ImGui::BeginTabItem("Tests"))
		{
			int categoryIndex = 0;
			const char* category = g_sampleEntries[categoryIndex].category;
			int i = 0;
			while (i < g_sampleCount)
			{
				bool categorySelected = strcmp(category, g_sampleEntries[s_settings.sampleIndex].category) == 0;
				ImGuiTreeNodeFlags nodeSelectionFlags = categorySelected ? ImGuiTreeNodeFlags_Selected : 0;
				bool nodeOpen = ImGui::TreeNodeEx(category, nodeFlags | nodeSelectionFlags);

				if (nodeOpen)
				{
					while (i < g_sampleCount && strcmp(category, g_sampleEntries[i].category) == 0)
					{
						ImGuiTreeNodeFlags selectionFlags = 0;
						if (s_settings.sampleIndex == i)
						{
							selectionFlags = ImGuiTreeNodeFlags_Selected;
						}
						ImGui::TreeNodeEx((void*)(intptr_t)i, leafNodeFlags | selectionFlags, "%s", g_sampleEntries[i].name);
						if (ImGui::IsItemClicked())
						{
							s_selection = i;
						}
						++i;
					}
					ImGui::TreePop();
				}
				else
				{
					while (i < g_sampleCount && strcmp(category, g_sampleEntries[i].category) == 0)
					{
						++i;
					}
				}

				if (i < g_sampleCount)
				{
					category = g_sampleEntries[i].category;
					categoryIndex = i;
				}
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}

	ImGui::End();

	for (int i = 0; i < s2_solverTypeCount; ++i)
	{
		if (s_samples[i] != nullptr)
		{
			s_samples[i]->UpdateUI();
		}
	}
}

// Draws a segment of one meter length for scale
void DrawScale()
{
	s2Color c = s2MakeColor(s2_colorRed2, 1.0f);
	s2Vec2 s1 = {5.0f, g_camera.m_height - 10.0f};
	s2Vec2 p1 = g_camera.ConvertScreenToWorld(s1);
	s2Vec2 p2 = {p1.x + 1.0f, p1.y};
	g_draw.DrawSegment(p1, p2, c);

	s2Vec2 t1 = {s1.x, s1.y + 3.0f};
	s2Vec2 t2 = {s1.x, s1.y - 3.0f};
	g_draw.DrawSegment(g_camera.ConvertScreenToWorld(t1), g_camera.ConvertScreenToWorld(t2), c);

	s2Vec2 s2 = g_camera.ConvertWorldToScreen(p2);
	t1 = {s2.x, s2.y + 3.0f};
	t2 = {s2.x, s2.y - 3.0f};
	g_draw.DrawSegment(g_camera.ConvertScreenToWorld(t1), g_camera.ConvertScreenToWorld(t2), c);
}

int main(int, char**)
{
#if defined(_WIN32)
	// Enable memory-leak reports
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

	char buffer[128];

	s_settings.Load();
	SortTests();

	glfwSetErrorCallback(glfwErrorCallback);

	g_camera.m_width = s_settings.windowWidth;
	g_camera.m_height = s_settings.windowHeight;

	if (glfwInit() == 0)
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

#if __APPLE__
	const char* glslVersion = "#version 150";
#else
	const char* glslVersion = nullptr;
#endif

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// MSAA
	glfwWindowHint(GLFW_SAMPLES, 4);

	snprintf(buffer, 128, "Solver2D");

	if (GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor())
	{
#ifdef __APPLE__
		glfwGetMonitorContentScale(primaryMonitor, &s_framebufferScale, &s_framebufferScale);
#else
		glfwGetMonitorContentScale(primaryMonitor, &s_windowScale, &s_windowScale);
#endif
	}

	bool fullscreen = false;
	if (fullscreen)
	{
		g_mainWindow =
			glfwCreateWindow(int(1920 * s_windowScale), int(1080 * s_windowScale), buffer, glfwGetPrimaryMonitor(), nullptr);
	}
	else
	{
		g_mainWindow = glfwCreateWindow(int(g_camera.m_width * s_windowScale), int(g_camera.m_height * s_windowScale), buffer,
										nullptr, nullptr);
	}

	if (g_mainWindow == nullptr)
	{
		fprintf(stderr, "Failed to open GLFW g_mainWindow.\n");
		glfwTerminate();
		return -1;
	}

#ifdef __APPLE__
	glfwGetWindowContentScale(g_mainWindow, &s_framebufferScale, &s_framebufferScale);
#else
	glfwGetWindowContentScale(g_mainWindow, &s_windowScale, &s_windowScale);
#endif

	glfwMakeContextCurrent(g_mainWindow);

	// Load OpenGL functions using glad
	if (!gladLoadGL())
	{
		fprintf(stderr, "Failed to initialize glad\n");
		glfwTerminate();
		return -1;
	}

	printf("GL %d.%d\n", GLVersion.major, GLVersion.minor);
	printf("OpenGL %s, GLSL %s\n", glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

	glfwSetWindowSizeCallback(g_mainWindow, ResizeWindowCallback);
	glfwSetKeyCallback(g_mainWindow, KeyCallback);
	glfwSetCharCallback(g_mainWindow, CharCallback);
	glfwSetMouseButtonCallback(g_mainWindow, MouseButtonCallback);
	glfwSetCursorPosCallback(g_mainWindow, MouseMotionCallback);
	glfwSetScrollCallback(g_mainWindow, ScrollCallback);

	g_draw.Create();
	CreateUI(g_mainWindow, glslVersion);

	s_settings.sampleIndex = S2_CLAMP(s_settings.sampleIndex, 0, g_sampleCount - 1);
	s_selection = s_settings.sampleIndex;

	float colorAlpha = 1.0f;
	s2Color solverColors[s2_solverTypeCount] = {
		s2MakeColor(s2_colorCyan, colorAlpha),
		s2MakeColor(s2_colorDodgerBlue, colorAlpha),
		s2MakeColor(s2_colorBlueViolet, colorAlpha),
		s2MakeColor(s2_colorCoral, colorAlpha),
		s2MakeColor(s2_colorLightBlue, colorAlpha),
		s2MakeColor(s2_colorLavenderBlush, colorAlpha),
		s2MakeColor(s2_colorYellow2, colorAlpha),
		s2MakeColor(s2_colorOrchid, colorAlpha),
		s2MakeColor(s2_colorSpringGreen, colorAlpha),
		s2MakeColor(s2_colorOrangeRed, colorAlpha),
	};

	static_assert(S2_ARRAY_COUNT(solverColors) == s2_solverTypeCount);

	for (int i = 0; i < s2_solverTypeCount; ++i)
	{
		if (s_settings.enabledSolvers[i])
		{
			s_samples[i] = g_sampleEntries[s_settings.sampleIndex].createFcn(s_settings, s2SolverType(i));
		}
	}

	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

	float frameTime = 0.0;
	int frame = 0;

	while (!glfwWindowShouldClose(g_mainWindow))
	{
		double time1 = glfwGetTime();
		s_settings.textLine = 0;

		if (glfwGetKey(g_mainWindow, GLFW_KEY_Z) == GLFW_PRESS)
		{
			// Zoom out
			g_camera.m_zoom = S2_MIN(1.005f * g_camera.m_zoom, 20.0f);
		}
		else if (glfwGetKey(g_mainWindow, GLFW_KEY_X) == GLFW_PRESS)
		{
			// Zoom in
			g_camera.m_zoom = S2_MAX(0.995f * g_camera.m_zoom, 0.02f);
		}
		else if (glfwGetKey(g_mainWindow, GLFW_KEY_O) == GLFW_PRESS)
		{
			s_settings.singleStep = true;
		}

		glfwGetWindowSize(g_mainWindow, &g_camera.m_width, &g_camera.m_height);
		g_camera.m_width = int(g_camera.m_width / s_windowScale);
		g_camera.m_height = int(g_camera.m_height / s_windowScale);

		int bufferWidth, bufferHeight;
		glfwGetFramebufferSize(g_mainWindow, &bufferWidth, &bufferHeight);
		glViewport(0, 0, bufferWidth, bufferHeight);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double cursorPosX = 0, cursorPosY = 0;
		glfwGetCursorPos(g_mainWindow, &cursorPosX, &cursorPosY);
		ImGui_ImplGlfw_CursorPosCallback(g_mainWindow, cursorPosX / s_windowScale, cursorPosY / s_windowScale);
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui_ImplGlfw_CursorPosCallback(g_mainWindow, cursorPosX / s_windowScale, cursorPosY / s_windowScale);

		ImGuiIO& io = ImGui::GetIO();
		io.DisplaySize.x = float(g_camera.m_width);
		io.DisplaySize.y = float(g_camera.m_height);
		io.DisplayFramebufferScale.x = bufferWidth / float(g_camera.m_width);
		io.DisplayFramebufferScale.y = bufferHeight / float(g_camera.m_height);

		ImGui::NewFrame();

		ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
		ImGui::SetNextWindowSize(ImVec2(float(g_camera.m_width), float(g_camera.m_height)));
		ImGui::SetNextWindowBgAlpha(0.0f);
		ImGui::Begin("Overlay", nullptr,
					 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize |
						 ImGuiWindowFlags_NoScrollbar);
		ImGui::End();

		if (g_draw.m_showUI)
		{
			const SampleEntry& entry = g_sampleEntries[s_settings.sampleIndex];
			snprintf(buffer, 128, "%s : %s", entry.category, entry.name);

			for (int i = 0; i < s2_solverTypeCount; ++i)
			{
				if (s_samples[i] != nullptr)
				{
					s_samples[i]->DrawTitle(s_settings, buffer);
					break;
				}
			}
		}

		s_settings.timeStep = s_settings.hertz > 0.0f ? 1.0f / s_settings.hertz : float(0.0f);

		if (s_settings.pause)
		{
			if (s_settings.singleStep)
			{
				s_settings.singleStep = 0;
			}
			else
			{
				s_settings.timeStep = 0.0f;
			}

			//g_draw.DrawString(5, s_settings.textLine, "****PAUSED****");
			//s_settings.textLine += s_settings.textIncrement;
		}

		g_draw.m_debugDraw.drawShapes = s_settings.drawShapes;
		g_draw.m_debugDraw.drawJoints = s_settings.drawJoints;
		g_draw.m_debugDraw.drawAABBs = s_settings.drawAABBs;
		g_draw.m_debugDraw.drawMass = s_settings.drawMass;
		g_draw.m_debugDraw.drawContactPoints = s_settings.drawContactPoints;
		g_draw.m_debugDraw.drawContactNormals = s_settings.drawContactNormals;
		g_draw.m_debugDraw.drawContactImpulses = s_settings.drawContactImpulse;
		g_draw.m_debugDraw.drawFrictionImpulses = s_settings.drawFrictionImpulse;

		int stepCount = 0;
		for (int i = 0; i < s2_solverTypeCount; ++i)
		{
			if (s_samples[i] != nullptr)
			{
				s_samples[i]->Step(s_settings, solverColors[i]);
				stepCount = s_samples[i]->m_stepCount;
			}
		}

		g_draw.Flush();

		UpdateUI(solverColors);

		// ImGui::ShowDemoWindow();

		// if (g_draw.m_showUI)
		{
			snprintf(buffer, 128, "%.1f ms - step %d - camera (%g, %g)", 1000.0f * frameTime, stepCount, g_camera.m_center.x, g_camera.m_center.y);
			ImGui::Begin("Overlay", nullptr,
						 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize |
							 ImGuiWindowFlags_NoScrollbar);
			ImGui::SetCursorPos(ImVec2(5.0f, g_camera.m_height - 30.0f));
			ImGui::TextColored(ImColor(153, 230, 153, 255), "%s", buffer);
			ImGui::End();
		}

		DrawScale();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(g_mainWindow);

		if (s_selection != s_settings.sampleIndex)
		{
			s_settings.sampleIndex = s_selection;
			for (int i = 0; i < s2_solverTypeCount; ++i)
			{
				if (s_samples[i] != nullptr)
				{
					delete s_samples[i];
					s_samples[i] = nullptr;
				}
			}

			s_settings.restart = false;
			g_camera.ResetView();

			for (int i = 0; i < s2_solverTypeCount; ++i)
			{
				if (s_settings.enabledSolvers[i])
				{
					s_samples[i] = g_sampleEntries[s_settings.sampleIndex].createFcn(s_settings, s2SolverType(i));
				}
			}
		}

		glfwPollEvents();

		// Limit frame rate to 60Hz
		double time2 = glfwGetTime();
		double targetTime = time1 + 1.0f / 60.0f;
		int loopCount = 0;
		while (time2 < targetTime)
		{
#if defined(_WIN32)
			Sleep(0.0f);
#endif
			time2 = glfwGetTime();
			++loopCount;
		}

		frameTime = (float)(time2 - time1);
		// if (frame % 17 == 0)
		//{
		//	printf("loop count = %d, frame time = %.1f\n", loopCount, 1000.0f * frameTime);
		// }
		++frame;
	}

	for (int i = 0; i < s2_solverTypeCount; ++i)
	{
		if (s_samples[i] != nullptr)
		{
			delete s_samples[i];
			s_samples[i] = nullptr;
		}
	}

	g_draw.Destroy();

	DestroyUI();
	glfwTerminate();

	s_settings.Save();

#if defined(_WIN32)
	_CrtDumpMemoryLeaks();
#endif

	return 0;
}
